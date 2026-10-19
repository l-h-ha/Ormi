#include <filesystem>
#include <string>
#include <string_view>

#include <dlfcn.h>

#include <ormi/core/error.h>

#include "platform/platform.h"

namespace ormi::platform {
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

    std::string_view shared_extension() {
        return ".so";
    }

    std::filesystem::path library_path(const void* anchor, std::string_view stem) {
        Dl_info module{};
        if (dladdr(anchor, &module) == 0 || !module.dli_fname) {
            ORMI_THROW(jitError, "Cannot locate the ORMI core module.");
        }

        const auto filename = "lib" + std::string(stem) + ".so";
        return std::filesystem::absolute(module.dli_fname).parent_path() / filename;
    }
}
