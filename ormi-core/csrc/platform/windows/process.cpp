#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <ormi/core/error.h>

#include "platform/platform.h"
#include "platform/windows/utf.h"

namespace ormi::platform {
    namespace {
        class Handle {
        public:
            Handle() = default;
            explicit Handle(HANDLE value) : value_(value) {}

            ~Handle() {
                reset();
            }

            Handle(const Handle&) = delete;
            Handle& operator=(const Handle&) = delete;

            Handle(Handle&& other) noexcept : value_(other.value_) {
                other.value_ = nullptr;
            }

            Handle& operator=(Handle&& other) noexcept {
                if (this != &other) {
                    reset();
                    value_ = other.value_;
                    other.value_ = nullptr;
                }
                return *this;
            }

            HANDLE get() const {
                return value_;
            }

            void reset(HANDLE value = nullptr) {
                if (value_ && value_ != INVALID_HANDLE_VALUE) {
                    CloseHandle(value_);
                }
                value_ = value;
            }

        private:
            HANDLE value_ = nullptr;
        };

        std::wstring quote(std::wstring_view argument) {
            if (argument.find_first_of(L" \t\"") == std::wstring_view::npos) {
                return std::wstring(argument);
            }

            std::wstring result(1, L'\"');
            std::size_t slashes = 0;
            for (const auto character : argument) {
                if (character == L'\\') {
                    ++slashes;
                    continue;
                }

                if (character == L'\"') {
                    result.append(slashes * 2 + 1, L'\\');
                    result.push_back(L'\"');
                } else {
                    result.append(slashes, L'\\');
                    result.push_back(character);
                }
                slashes = 0;
            }

            result.append(slashes * 2, L'\\');
            result.push_back(L'\"');
            return result;
        }

        std::vector<wchar_t> environment_block(
            std::span<const std::string> environment
        ) {
            std::vector<wchar_t> result;
            for (const auto& variable : environment) {
                const auto wide = windows::widen(variable);
                result.insert(result.end(), wide.begin(), wide.end());
                result.push_back(L'\0');
            }
            result.push_back(L'\0');
            return result;
        }
    }

    std::optional<std::filesystem::path> find(std::string_view executable) {
        const std::filesystem::path requested(windows::widen(executable));
        if (requested.has_parent_path() || requested.is_absolute()) {
            if (std::filesystem::is_regular_file(requested)) {
                return std::filesystem::absolute(requested);
            }
            return std::nullopt;
        }

        std::vector<wchar_t> buffer(32768);
        const auto size = SearchPathW(
            nullptr,
            requested.c_str(),
            nullptr,
            static_cast<DWORD>(buffer.size()),
            buffer.data(),
            nullptr
        );
        if (size == 0 || size >= buffer.size()) {
            return std::nullopt;
        }
        return std::filesystem::path(buffer.data());
    }

    Process run(
        const std::filesystem::path& executable,
        std::span<const std::string> arguments,
        std::span<const std::string> environment,
        std::chrono::milliseconds timeout
    ) {
        SECURITY_ATTRIBUTES security{
            sizeof(SECURITY_ATTRIBUTES),
            nullptr,
            TRUE
        };
        HANDLE raw_read = nullptr;
        HANDLE raw_write = nullptr;
        if (!CreatePipe(&raw_read, &raw_write, &security, 0)) {
            ORMI_THROW(jitError, "Cannot create compiler output pipe.");
        }

        Handle read(raw_read);
        Handle write(raw_write);
        if (!SetHandleInformation(read.get(), HANDLE_FLAG_INHERIT, 0)) {
            ORMI_THROW(jitError, "Cannot configure compiler output pipe.");
        }

        std::wstring command = quote(executable.wstring());
        for (const auto& argument : arguments) {
            command.push_back(L' ');
            command += quote(windows::widen(argument));
        }

        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = write.get();
        startup.hStdError = write.get();
        PROCESS_INFORMATION process{};
        auto configured_environment = environment.empty()
            ? std::vector<wchar_t>{}
            : environment_block(environment);
        const DWORD creation_flags = CREATE_NO_WINDOW | CREATE_SUSPENDED |
            (environment.empty() ? 0 : CREATE_UNICODE_ENVIRONMENT);

        Handle job(CreateJobObjectW(nullptr, nullptr));
        if (!job.get()) {
            ORMI_THROW(jitError, "Cannot create compiler process job.");
        }
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info{};
        job_info.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(
            job.get(),
            JobObjectExtendedLimitInformation,
            &job_info,
            sizeof(job_info)
        )) {
            ORMI_THROW(jitError, "Cannot configure compiler process job.");
        }

        if (!CreateProcessW(
            executable.c_str(),
            command.data(),
            nullptr,
            nullptr,
            TRUE,
            creation_flags,
            configured_environment.empty()
                ? nullptr
                : configured_environment.data(),
            nullptr,
            &startup,
            &process
        )) {
            ORMI_THROW(
                jitError,
                "Cannot start compiler executable: " + executable.string()
            );
        }

        Handle process_handle(process.hProcess);
        Handle thread_handle(process.hThread);
        if (!AssignProcessToJobObject(job.get(), process_handle.get())) {
            TerminateProcess(process_handle.get(), ERROR_PROCESS_ABORTED);
            WaitForSingleObject(process_handle.get(), INFINITE);
            ORMI_THROW(jitError, "Cannot assign compiler process to its job.");
        }
        if (ResumeThread(thread_handle.get()) == static_cast<DWORD>(-1)) {
            TerminateJobObject(job.get(), ERROR_PROCESS_ABORTED);
            WaitForSingleObject(process_handle.get(), INFINITE);
            ORMI_THROW(jitError, "Cannot resume compiler process.");
        }

        write.reset();
        std::string output;
        std::array<char, 4096> bytes{};
        const bool bounded = timeout != (std::chrono::milliseconds::max)();
        const auto deadline = std::chrono::steady_clock::now() +
            (bounded ? timeout : std::chrono::milliseconds{0});
        bool timed_out = false;

        const auto drain = [&]() {
            while (true) {
                DWORD available = 0;
                if (!PeekNamedPipe(
                    read.get(),
                    nullptr,
                    0,
                    nullptr,
                    &available,
                    nullptr
                )) {
                    if (GetLastError() == ERROR_BROKEN_PIPE) {
                        return;
                    }
                    ORMI_THROW(
                        jitError,
                        "Cannot inspect compiler process output."
                    );
                }
                if (available == 0) {
                    return;
                }

                DWORD read_bytes = 0;
                const auto requested = static_cast<DWORD>(
                    std::min<std::size_t>(bytes.size(), available)
                );
                if (!ReadFile(
                    read.get(),
                    bytes.data(),
                    requested,
                    &read_bytes,
                    nullptr
                )) {
                    if (GetLastError() == ERROR_BROKEN_PIPE) {
                        return;
                    }
                    ORMI_THROW(jitError, "Cannot read compiler process output.");
                }
                output.append(bytes.data(), read_bytes);
            }
        };

        while (true) {
            drain();
            const auto state = WaitForSingleObject(process_handle.get(), 0);
            if (state == WAIT_OBJECT_0) {
                break;
            }
            if (state == WAIT_FAILED) {
                ORMI_THROW(jitError, "Cannot wait for compiler process.");
            }

            if (bounded && std::chrono::steady_clock::now() >= deadline) {
                timed_out = true;
                TerminateJobObject(job.get(), ERROR_TIMEOUT);
                WaitForSingleObject(process_handle.get(), INFINITE);
                break;
            }

            DWORD wait = 10;
            if (bounded) {
                const auto remaining = std::chrono::duration_cast<
                    std::chrono::milliseconds
                >(deadline - std::chrono::steady_clock::now());
                wait = static_cast<DWORD>(std::clamp<long long>(
                    remaining.count(),
                    1,
                    10
                ));
            }
            WaitForSingleObject(process_handle.get(), wait);
        }

        drain();
        DWORD code = 0;
        if (!GetExitCodeProcess(process_handle.get(), &code)) {
            ORMI_THROW(jitError, "Cannot read compiler process exit code.");
        }

        const auto state = timed_out
            ? Process::State::timed_out
            : code >= 0xc0000000ul
                ? Process::State::crashed
                : Process::State::exited;
        return Process{static_cast<int>(code), std::move(output), state};
    }

    unsigned long long process_id() {
        return static_cast<unsigned long long>(GetCurrentProcessId());
    }
}
