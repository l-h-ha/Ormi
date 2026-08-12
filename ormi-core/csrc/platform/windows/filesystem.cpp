#include <filesystem>
#include <string>
#include <system_error>
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
            const auto code = GetLastError();
            ORMI_THROW(
                jitError,
                "Cannot open cache lock: " + path.string() + "\n" +
                    std::system_category().message(static_cast<int>(code)) +
                    " (" + std::to_string(code) + ")"
            );
        }

        OVERLAPPED overlap{};
        const auto flags = mode == Lock::exclusive
            ? LOCKFILE_EXCLUSIVE_LOCK
            : 0;
        if (!LockFileEx(
            handle,
            flags,
            0,
            MAXDWORD,
            MAXDWORD,
            &overlap
        )) {
            CloseHandle(handle);
            ORMI_THROW(
                jitError,
                "Cannot lock cache entry: " + path.string()
            );
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

    std::filesystem::path cache_directory() {
        const auto local = environment_path(L"LOCALAPPDATA");
        return local.empty()
            ? std::filesystem::temp_directory_path() / "ormi-cache"
            : local / "ormi";
    }
}
