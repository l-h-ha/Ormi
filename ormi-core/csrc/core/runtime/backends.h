#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include <ormi/backend/api.h>

#include "core/runtime/catalog.h"

namespace ormi::core::runtime {
    class Backends {
    public:
        static Backends& instance();

        ~Backends();

        Backends(const Backends&) = delete;
        Backends& operator=(const Backends&) = delete;

        BackendId find(std::string_view name) const;
        bool available(BackendId id) const;
        const Backend& require(BackendId id);

    private:
        struct Plugin;

        Backends();

        void init(BackendId id);
        void install(
            BackendId id,
            Backend backend,
            std::unique_ptr<Plugin> plugin = nullptr
        );

        std::array<std::optional<Backend>, catalog::entries.size()> backends_;
        std::array<std::atomic<const Backend*>, catalog::entries.size()> resolved_{};
        std::array<std::unique_ptr<Plugin>, catalog::entries.size()> plugins_;
        std::array<std::string, catalog::entries.size()> errors_;
        mutable std::mutex mutex_;
    };
}
