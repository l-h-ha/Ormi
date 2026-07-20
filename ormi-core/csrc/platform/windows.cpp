#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <ormi/core/error.h>

#include "platform/platform.h"

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

        std::wstring widen(std::string_view text) {
            if (text.empty()) {
                return {};
            }

            const auto size = MultiByteToWideChar(
                CP_UTF8,
                MB_ERR_INVALID_CHARS,
                text.data(),
                static_cast<int>(text.size()),
                nullptr,
                0
            );

            if (size <= 0) {
                ORMI_THROW(jitError, "Cannot convert UTF-8 text to UTF-16.");
            }

            std::wstring result(static_cast<std::size_t>(size), L'\0');
            MultiByteToWideChar(
                CP_UTF8,
                MB_ERR_INVALID_CHARS,
                text.data(),
                static_cast<int>(text.size()),
                result.data(),
                size
            );
            return result;
        }

        std::string trim(std::string value) {
            while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
                value.pop_back();
            }
            const auto first = std::find_if_not(
                value.begin(),
                value.end(),
                [](unsigned char character) { return std::isspace(character); }
            );
            value.erase(value.begin(), first);
            return value;
        }

        std::filesystem::path environment_path(const wchar_t* name) {
            std::vector<wchar_t> value(32768);
            const auto size = GetEnvironmentVariableW(
                name,
                value.data(),
                static_cast<DWORD>(value.size())
            );
            if (size == 0 || size >= value.size()) {
                return {};
            }
            return std::filesystem::path(value.data(), value.data() + size);
        }

        std::filesystem::path short_path(const std::filesystem::path& path) {
            std::vector<wchar_t> value(32768);
            const auto size = GetShortPathNameW(
                path.c_str(),
                value.data(),
                static_cast<DWORD>(value.size())
            );
            if (size == 0 || size >= value.size()) {
                return {};
            }
            return std::filesystem::path(value.data(), value.data() + size);
        }

        std::vector<wchar_t> environment_block(
            std::span<const std::string> environment
        ) {
            std::vector<wchar_t> result;
            for (const auto& variable : environment) {
                const auto wide = widen(variable);
                result.insert(result.end(), wide.begin(), wide.end());
                result.push_back(L'\0');
            }
            result.push_back(L'\0');
            return result;
        }

        std::vector<std::string> parse_environment(std::string_view output) {
            std::vector<std::string> result;
            std::unordered_set<std::string> keys;
            while (!output.empty()) {
                const auto newline = output.find('\n');
                auto line = output.substr(0, newline);
                if (!line.empty() && line.back() == '\r') {
                    line.remove_suffix(1);
                }
                const auto separator = line.empty()
                    ? std::string_view::npos
                    : line.find('=', line.front() == '=' ? 1 : 0);
                if (separator != std::string_view::npos) {
                    std::string key(line.substr(0, separator));
                    std::transform(
                        key.begin(),
                        key.end(),
                        key.begin(),
                        [](unsigned char character) {
                            return static_cast<char>(std::tolower(character));
                        }
                    );

                    if (keys.insert(std::move(key)).second) {
                        result.emplace_back(line);
                    }
                }
                if (newline == std::string_view::npos) {
                    break;
                }
                output.remove_prefix(newline + 1);
            }
            return result;
        }

        std::optional<std::string_view> environment_value(
            const std::vector<std::string>& environment,
            std::string_view requested
        ) {
            for (const auto& variable : environment) {
                const auto separator = variable.find('=');
                if (separator != requested.size()) {
                    continue;
                }

                bool equal = true;
                for (std::size_t i = 0; i < requested.size(); ++i) {
                    const auto lhs = static_cast<unsigned char>(variable[i]);
                    const auto rhs = static_cast<unsigned char>(requested[i]);
                    if (std::tolower(lhs) != std::tolower(rhs)) {
                        equal = false;
                        break;
                    }
                }
                if (equal) {
                    return std::string_view(variable).substr(separator + 1);
                }
            }
            return std::nullopt;
        }
    }

    std::optional<std::filesystem::path> find(std::string_view executable) {
        const std::filesystem::path requested(widen(executable));

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

    FileLock::FileLock(const std::filesystem::path& path, Lock mode) {
        const auto handle = CreateFileW(
            path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (handle == INVALID_HANDLE_VALUE) {
            ORMI_THROW(jitError, "Cannot open CPU JIT cache lock: " + path.string());
        }

        OVERLAPPED overlap{};
        const auto flags = mode == Lock::exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0;
        if (!LockFileEx(
            handle,
            flags,
            0,
            MAXDWORD,
            MAXDWORD,
            &overlap
        )) {
            CloseHandle(handle);
            ORMI_THROW(jitError, "Cannot lock CPU JIT cache entry: " + path.string());
        }
        handle_ = reinterpret_cast<std::intptr_t>(handle);
    }

    FileLock::~FileLock() {
        if (handle_ == -1) {
            return;
        }

        const auto handle = reinterpret_cast<HANDLE>(handle_);
        OVERLAPPED overlap{};
        UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &overlap);
        CloseHandle(handle);
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
            command += quote(widen(argument));
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
        job_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
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
            configured_environment.empty() ? nullptr : configured_environment.data(),
            nullptr,
            &startup,
            &process
        )) {
            ORMI_THROW(jitError, "Cannot start compiler executable: " + executable.string());
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
                    ORMI_THROW(jitError, "Cannot inspect compiler process output.");
                }
                if (available == 0) {
                    return;
                }

                DWORD read_bytes = 0;
                const auto requested = static_cast<DWORD>(std::min<std::size_t>(
                    bytes.size(),
                    available
                ));
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

    std::optional<Tool> configured_msvc() {
        static const auto configured = []() -> std::optional<Tool> {
            const auto program_files = environment_path(L"ProgramFiles(x86)");
            if (program_files.empty()) {
                return std::nullopt;
            }

            const auto vswhere = program_files /
                "Microsoft Visual Studio" /
                "Installer" /
                "vswhere.exe";
            if (!std::filesystem::is_regular_file(vswhere)) {
                return std::nullopt;
            }

            const auto installation = run(
                vswhere,
                std::vector<std::string>{
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath"
                }
            );
            if (installation.code != 0) {
                return std::nullopt;
            }

            const auto root = std::filesystem::path(widen(trim(installation.output)));
            const auto setup = root / "Common7" / "Tools" / "VsDevCmd.bat";
            const auto shortened = short_path(setup);
            const auto command = find("cmd.exe");
            if (
                root.empty() ||
                shortened.empty() ||
                !std::filesystem::is_regular_file(setup) ||
                !command
            ) {
                return std::nullopt;
            }

            const auto environment_result = run(
                *command,
                std::vector<std::string>{
                    "/d",
                    "/s",
                    "/c",
                    "call " + shortened.string() +
                        " -arch=x64 -host_arch=x64 >nul && set"
                }
            );
            if (environment_result.code != 0) {
                return std::nullopt;
            }

            auto environment = parse_environment(environment_result.output);
            const auto tools = environment_value(environment, "VCToolsInstallDir");
            if (!tools) {
                return std::nullopt;
            }

            const auto compiler = std::filesystem::path(widen(*tools)) /
                "bin" /
                "Hostx64" /
                "x64" /
                "cl.exe";
            if (!std::filesystem::is_regular_file(compiler)) {
                return std::nullopt;
            }
            return Tool{compiler, std::move(environment)};
        }();

        return configured;
    }

    Library::Library(const std::filesystem::path& path)
        : handle_(reinterpret_cast<void*>(LoadLibraryExW(
            path.c_str(),
            nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
        )))
    {
        if (!handle_) {
            const auto code = GetLastError();
            ORMI_THROW(
                jitError,
                "Cannot load shared library: " + path.string() +
                    " (Windows error " + std::to_string(code) + ")."
            );
        }
    }

    Library::~Library() {
        if (handle_) {
            FreeLibrary(reinterpret_cast<HMODULE>(handle_));
        }
    }

    Library::Library(Library&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Library& Library::operator=(Library&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                FreeLibrary(reinterpret_cast<HMODULE>(handle_));
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void* Library::symbol(std::string_view name) const {
        const std::string owned(name);
        auto* result = reinterpret_cast<void*>(
            GetProcAddress(reinterpret_cast<HMODULE>(handle_), owned.c_str())
        );
        if (!result) {
            ORMI_THROW(jitError, "Shared library does not export symbol: " + owned);
        }
        return result;
    }

    std::span<const std::string_view> cxx_candidates() {
        static constexpr std::array names{
            std::string_view{"cl.exe"},
            std::string_view{"clang-cl.exe"},
            std::string_view{"g++.exe"},
            std::string_view{"clang++.exe"}
        };
        return names;
    }

    std::vector<std::string> shared_flags() {
        return {"-shared", "-static-libgcc", "-static-libstdc++"};
    }

    std::string_view shared_extension() {
        return ".dll";
    }

    std::filesystem::path cache_directory() {
        const auto local = environment_path(L"LOCALAPPDATA");
        return local.empty()
            ? std::filesystem::temp_directory_path() / "ormi-cache"
            : local / "ormi";
    }

    std::filesystem::path library_path(const void* anchor, std::string_view stem) {
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(anchor),
            &module
        )) {
            ORMI_THROW(jitError, "Cannot locate the ORMI core module.");
        }

        std::vector<wchar_t> path(32768);
        const auto size = GetModuleFileNameW(
            module,
            path.data(),
            static_cast<DWORD>(path.size())
        );
        if (size == 0 || size >= path.size()) {
            ORMI_THROW(jitError, "Cannot locate the ORMI core module path.");
        }

        const auto filename = std::string(stem) + ".dll";
        return std::filesystem::path(path.data(), path.data() + size).parent_path() /
            std::filesystem::path(widen(filename));
    }

    unsigned long long process_id() {
        return static_cast<unsigned long long>(GetCurrentProcessId());
    }
}
