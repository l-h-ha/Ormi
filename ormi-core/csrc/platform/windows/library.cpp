#include <filesystem>
#include <string>
#include <string_view>
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
        HMODULE load_library(const std::filesystem::path& path) {
            const auto absolute = path.is_absolute()
                ? path
                : std::filesystem::absolute(path);
            auto* handle = LoadLibraryExW(
                absolute.c_str(),
                nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
            );
            if (!handle) {
                handle = LoadLibraryExW(
                    absolute.c_str(),
                    nullptr,
                    LOAD_WITH_ALTERED_SEARCH_PATH
                );
            }
            return handle;
        }
    }

    Library::Library(const std::filesystem::path& path)
        : handle_(reinterpret_cast<void*>(load_library(path))) {
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
        auto* result = reinterpret_cast<void*>(GetProcAddress(
            reinterpret_cast<HMODULE>(handle_),
            owned.c_str()
        ));
        if (!result) {
            ORMI_THROW(
                jitError,
                "Shared library does not export symbol: " + owned
            );
        }
        return result;
    }

    std::string_view shared_extension() {
        return ".dll";
    }

    std::filesystem::path library_path(
        const void* anchor,
        std::string_view stem
    ) {
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
        return std::filesystem::path(
            path.data(),
            path.data() + size
        ).parent_path() / std::filesystem::path(windows::widen(filename));
    }
}
