#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <ormi/core/error.h>

#include "platform/platform.h"

namespace ormi::platform {
    Process run(
        const std::filesystem::path& executable,
        std::span<const std::string> arguments,
        std::span<const std::string> environment,
        std::chrono::milliseconds timeout
    ) {
        std::vector<std::string> storage;
        storage.reserve(arguments.size() + 1);
        storage.push_back(executable.string());
        storage.insert(storage.end(), arguments.begin(), arguments.end());

        std::vector<char*> argv;
        argv.reserve(storage.size() + 1);
        for (auto& argument : storage) {
            argv.push_back(argument.data());
        }
        argv.push_back(nullptr);

        std::vector<char*> envp;
        if (!environment.empty()) {
            envp.reserve(environment.size() + 1);
            for (const auto& variable : environment) {
                envp.push_back(const_cast<char*>(variable.c_str()));
            }
            envp.push_back(nullptr);
        }

        int output_pipe[2];
        if (pipe(output_pipe) != 0) {
            ORMI_THROW(jitError, "Cannot create compiler output pipe.");
        }

        const auto child = fork();
        if (child < 0) {
            close(output_pipe[0]);
            close(output_pipe[1]);
            ORMI_THROW(jitError, "Cannot fork compiler process.");
        }

        if (child == 0) {
            setpgid(0, 0);
            close(output_pipe[0]);
            dup2(output_pipe[1], STDOUT_FILENO);
            dup2(output_pipe[1], STDERR_FILENO);
            close(output_pipe[1]);

            if (environment.empty()) {
                execv(executable.c_str(), argv.data());
            } else {
                execve(executable.c_str(), argv.data(), envp.data());
            }
            _exit(127);
        }

        close(output_pipe[1]);
        if (setpgid(child, child) != 0 && errno != EACCES && errno != ESRCH) {
            kill(child, SIGKILL);
            waitpid(child, nullptr, 0);
            close(output_pipe[0]);
            ORMI_THROW(jitError, "Cannot isolate compiler process group.");
        }

        const auto flags = fcntl(output_pipe[0], F_GETFL, 0);
        if (flags < 0 || fcntl(output_pipe[0], F_SETFL, flags | O_NONBLOCK) != 0) {
            kill(-child, SIGKILL);
            kill(child, SIGKILL);
            waitpid(child, nullptr, 0);
            close(output_pipe[0]);
            ORMI_THROW(jitError, "Cannot configure compiler output pipe.");
        }

        std::string output;
        std::array<char, 4096> bytes{};
        int status = 0;
        bool finished = false;
        bool timed_out = false;
        const bool bounded = timeout != (std::chrono::milliseconds::max)();
        const auto deadline = std::chrono::steady_clock::now() +
            (bounded ? timeout : std::chrono::milliseconds{0});

        const auto terminate = [&] {
            kill(-child, SIGKILL);
            kill(child, SIGKILL);
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
            finished = true;
        };

        const auto drain = [&]() {
            while (true) {
                const auto count = read(output_pipe[0], bytes.data(), bytes.size());
                if (count > 0) {
                    output.append(bytes.data(), static_cast<std::size_t>(count));
                    continue;
                }
                if (count == 0 || errno == EAGAIN || errno == EWOULDBLOCK) {
                    return;
                }
                if (errno != EINTR) {
                    terminate();
                    close(output_pipe[0]);
                    ORMI_THROW(jitError, "Cannot read compiler process output.");
                }
            }
        };

        while (!finished) {
            drain();
            const auto waited = waitpid(child, &status, WNOHANG);
            if (waited == child) {
                finished = true;
                break;
            }
            if (waited < 0 && errno != EINTR) {
                terminate();
                close(output_pipe[0]);
                ORMI_THROW(jitError, "Cannot wait for compiler process.");
            }

            if (bounded && std::chrono::steady_clock::now() >= deadline) {
                timed_out = true;
                terminate();
                break;
            }

            auto wait = 50;
            if (bounded) {
                const auto remaining = std::chrono::duration_cast<
                    std::chrono::milliseconds
                >(deadline - std::chrono::steady_clock::now());
                wait = static_cast<int>(std::clamp<long long>(
                    remaining.count(),
                    1,
                    50
                ));
            }
            pollfd descriptor{output_pipe[0], POLLIN | POLLHUP, 0};
            if (poll(&descriptor, 1, wait) < 0 && errno != EINTR) {
                terminate();
                close(output_pipe[0]);
                ORMI_THROW(jitError, "Cannot poll compiler process output.");
            }
        }

        drain();
        close(output_pipe[0]);
        const auto state = timed_out
            ? Process::State::timed_out
            : WIFSIGNALED(status)
                ? Process::State::crashed
                : Process::State::exited;
        const auto code = WIFEXITED(status)
            ? WEXITSTATUS(status)
            : WIFSIGNALED(status) ? WTERMSIG(status) : 128;
        return Process{code, std::move(output), state};
    }
}
