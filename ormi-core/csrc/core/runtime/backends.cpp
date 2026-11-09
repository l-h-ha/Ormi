#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <ormi/core/error.h>
#include <ormi/backend/plugin.h>

#include "core/runtime/backend.h"
#include "core/runtime/backends.h"
#include "platform/platform.h"

namespace ormi::core::runtime {
    namespace {
        const char core_anchor = 0;

        const catalog::Entry& entry(BackendId id) {
            if (id >= catalog::entries.size()) {
                ORMI_THROW(DeviceError, "Invalid backend id.");
            }
            return catalog::entries[id];
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

        for (std::size_t id = 0; id < catalog::entries.size(); ++id) {
            const auto& entry = catalog::entries[id];
            if (entry.init) {
                init(static_cast<BackendId>(id));
            }
        }
    }

    Backends::~Backends() = default;

    BackendId Backends::find(std::string_view name) const {
        return catalog::find(name);
    }

    bool Backends::available(BackendId id) const {
        return id < catalog::entries.size() &&
            resolved_[id].load(std::memory_order_acquire) != nullptr;
    }

    const Backend& Backends::require(BackendId id) {
        const auto& expected = entry(id);

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
            init(id);
        } catch (const OrmiError& error) {
            errors_[id] =
                "Cannot load backend '" + std::string(expected.name) +
                "'.\n" + error.what();
            ORMI_THROW(DeviceError, errors_[id]);
        }

        const auto* backend = resolved_[id].load(std::memory_order_relaxed);
        if (!backend) {
            ORMI_THROW(DeviceError, "Backend is not available.");
        }
        return *backend;
    }

    void Backends::init(BackendId id) {
        const auto& expected = entry(id);

        if (expected.init) {
            install(id, expected.init());
            return;
        }

        auto loaded = std::make_unique<Plugin>(
            platform::library_path(&core_anchor, expected.library)
        );
        const auto plugin_init = reinterpret_cast<plugin::Init>(
            loaded->library.symbol(plugin::entry)
        );
        const auto* backend = plugin_init();
        if (!backend) {
            ORMI_THROW(
                DeviceError,
                "Backend initialization returned no function table."
            );
        }
        if (backend->abi != backend_abi) {
            ORMI_THROW(DeviceError, "Backend interface version does not match this ORMI core.");
        }

        install(id, *backend, std::move(loaded));
    }

    void Backends::install(
        BackendId id,
        Backend backend,
        std::unique_ptr<Plugin> plugin
    ) {
        backend = normalize(std::move(backend), entry(id).name);

        backends_[id] = std::move(backend);
        plugins_[id] = std::move(plugin);
        resolved_[id].store(
            &*backends_[id],
            std::memory_order_release
        );
    }
}
