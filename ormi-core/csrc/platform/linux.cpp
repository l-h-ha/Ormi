#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <ormi/core/error.h>

#include "platform/platform.h"

namespace ormi::platform {
    namespace {
        bool executable(const std::filesystem::path& path) {
            return access(path.c_str(), X_OK) == 0;
        }
    }

    std::optional<std::filesystem::path> find(std::string_view name) {
        const std::filesystem::path requested(name);

        if (requested.has_parent_path() || requested.is_absolute()) {
            return executable(requested)
                ? std::optional<std::filesystem::path>(std::filesystem::absolute(requested))
                : std::nullopt;
        }

        const char* path = std::getenv("PATH");
        if (!path) {
            return std::nullopt;
        }

        std::string_view remaining(path);
        while (true) {
            const auto separator = remaining.find(':');
            const auto directory = remaining.substr(0, separator);
            const auto candidate =
                std::filesystem::path(directory.empty() ? "." : directory) /
                requested;

            if (executable(candidate)) {
                return std::filesystem::absolute(candidate);
            }

            if (separator == std::string_view::npos) {
                break;
            }
            remaining.remove_prefix(separator + 1);
        }

        return std::nullopt;
    }

    FileLock::FileLock(const std::filesystem::path& path, Lock mode) {
        const auto handle = open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (handle < 0) {
            ORMI_THROW(jitError, "Cannot open CPU JIT cache lock: " + path.string());
        }

        const auto operation = mode == Lock::exclusive ? LOCK_EX : LOCK_SH;
        while (flock(handle, operation) != 0) {
            if (errno != EINTR) {
                close(handle);
                ORMI_THROW(
                    jitError,
                    "Cannot lock CPU JIT cache entry: " + path.string()
                );
            }
        }
        handle_ = handle;
    }

    FileLock::~FileLock() {
        if (handle_ == -1) {
            return;
        }

        const auto handle = static_cast<int>(handle_);
        flock(handle, LOCK_UN);
        close(handle);
    }

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

    std::optional<Tool> configured_msvc() {
        return std::nullopt;
    }

    Library::Library(const std::filesystem::path& path)
        : handle_(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL))
    {
        if (!handle_) {
            const char* error = dlerror();
            ORMI_THROW(
                jitError,
                "Cannot load shared library: " + path.string() +
                (error ? "\n" + std::string(error) : "")
            );
        }
    }

    Library::~Library() {
        if (handle_) {
            dlclose(handle_);
        }
    }

    Library::Library(Library&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Library& Library::operator=(Library&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                dlclose(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void* Library::symbol(std::string_view name) const {
        const std::string owned(name);
        dlerror();
        auto* result = dlsym(handle_, owned.c_str());
        const char* error = dlerror();
        if (error) {
            ORMI_THROW(jitError, "Shared library does not export symbol: " + owned + "\n" + error);
        }
        return result;
    }

    std::span<const std::string_view> cxx_candidates() {
        static constexpr std::array names{
            std::string_view{"g++"},
            std::string_view{"clang++"},
            std::string_view{"c++"}
        };
        return names;
    }

    std::vector<std::string> shared_flags() {
        return {"-shared", "-fPIC"};
    }

    std::string_view shared_extension() {
        return ".so";
    }

    std::filesystem::path cache_directory() {
        if (const char* configured = std::getenv("XDG_CACHE_HOME")) {
            if (*configured) {
                return std::filesystem::path(configured) / "ormi";
            }
        }
        if (const char* home = std::getenv("HOME")) {
            if (*home) {
                return std::filesystem::path(home) / ".cache" / "ormi";
            }
        }
        return std::filesystem::temp_directory_path() / "ormi-cache";
    }

    std::filesystem::path library_path(const void* anchor, std::string_view stem) {
        Dl_info module{};
        if (dladdr(anchor, &module) == 0 || !module.dli_fname) {
            ORMI_THROW(jitError, "Cannot locate the ORMI core module.");
        }

        const auto filename = "lib" + std::string(stem) + ".so";
        return std::filesystem::absolute(module.dli_fname).parent_path() / filename;
    }

    unsigned long long process_id() {
        return static_cast<unsigned long long>(getpid());
    }
}
