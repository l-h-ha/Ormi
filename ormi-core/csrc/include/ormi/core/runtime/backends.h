#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include <ormi/core/runtime/backend.h>

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

        void init_cpu();
        void load_cuda();

        std::array<std::optional<Backend>, backend_count> backends_;
        std::array<std::atomic<const Backend*>, backend_count> resolved_{};
        std::array<std::unique_ptr<Plugin>, backend_count> plugins_;
        std::array<std::string, backend_count> errors_;
        mutable std::mutex mutex_;
    };
}
