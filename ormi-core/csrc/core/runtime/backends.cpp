#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <ormi/core/error.h>
#include <ormi/core/runtime/backends.h>
#include <ormi/core/runtime/plugin.h>

#include "backends/cpu/backend.h"
#include "platform/platform.h"

namespace ormi::core::runtime {
    namespace {
        const char core_anchor = 0;

        void validate(const Backend& backend, BackendId id) {
            if (backend.abi != backend_abi) {
                ORMI_THROW(DeviceError, "Backend ABI is incompatible with this ORMI core.");
            }

            if (backend.id != id) {
                ORMI_THROW(DeviceError, "Backend returned the wrong id.");
            }

            if (
                backend.name.empty() ||
                !backend.allocator ||
                !backend.copy ||
                !backend.fill ||
                !backend.fill_scalar ||
                !backend.cast ||
                !backend.sgd ||
                !backend.matmul ||
                !backend.reduce ||
                !backend.all_finite ||
                !backend.pointwise_binary ||
                !backend.pointwise_unary ||
                !backend.synchronize
            ) {
                ORMI_THROW(
                    DeviceError,
                    std::string(backend.name) + " backend returned an incomplete function table."
                );
            }
        }
    }

    struct Backends::Plugin {
        explicit Plugin(const std::filesystem::path& path) : library(path) {}

        platform::Library library;
    };

    Backends& Backends::instance() {
        static Backends backends;
        return backends;
    }

    Backends::Backends() {
        for (auto& backend : resolved_) {
            backend.store(nullptr, std::memory_order_relaxed);
        }

        init_cpu();
    }

    Backends::~Backends() = default;

    BackendId Backends::find(std::string_view name) const {
        if (name == "cpu") {
            return backend_id_cpu;
        }
        if (name == "cuda") {
            return backend_id_cuda;
        }
        return backend_invalid;
    }

    bool Backends::available(BackendId id) const {
        return id < backend_count &&
            resolved_[id].load(std::memory_order_acquire) != nullptr;
    }

    const Backend& Backends::require(BackendId id) {
        if (id >= backend_count) {
            ORMI_THROW(DeviceError, "Invalid backend id.");
        }

        if (const auto* backend = resolved_[id].load(std::memory_order_acquire)) {
            return *backend;
        }

        std::lock_guard lock(mutex_);

        if (const auto* backend = resolved_[id].load(std::memory_order_relaxed)) {
            return *backend;
        }

        if (!errors_[id].empty()) {
            ORMI_THROW(DeviceError, errors_[id]);
        }

        try {
            if (id == backend_id_cuda) {
                load_cuda();
            }
        } catch (const OrmiError& error) {
            errors_[id] = "Cannot load backend 'cuda'.\n" + std::string(error.what());
            ORMI_THROW(DeviceError, errors_[id]);
        }

        const auto* backend = resolved_[id].load(std::memory_order_relaxed);
        if (!backend) {
            ORMI_THROW(DeviceError, "Backend is not available.");
        }
        return *backend;
    }

    void Backends::init_cpu() {
        const auto backend = ormi::backends::cpu::init();
        validate(backend, backend_id_cpu);

        backends_[backend_id_cpu] = backend;
        resolved_[backend_id_cpu].store(
            &*backends_[backend_id_cpu],
            std::memory_order_release
        );
    }

    void Backends::load_cuda() {
        auto loaded = std::make_unique<Plugin>(
            platform::library_path(&core_anchor, "ormi_cuda_backend")
        );
        const auto init = reinterpret_cast<plugin::Init>(
            loaded->library.symbol(plugin::entry)
        );
        const auto* backend = init();
        if (!backend) {
            ORMI_THROW(DeviceError, "CUDA backend initialization returned no function table.");
        }

        validate(*backend, backend_id_cuda);
        backends_[backend_id_cuda] = *backend;
        plugins_[backend_id_cuda] = std::move(loaded);
        resolved_[backend_id_cuda].store(
            &*backends_[backend_id_cuda],
            std::memory_order_release
        );
    }
}
