#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backends/cuda/fusion/cache.h"
#include "backends/cuda/fusion/codegen.h"
#include "backends/cuda/fusion/compile.h"
#include "backends/cuda/fusion/module.h"
#include "backends/cuda/fusion/nvrtc.h"
#include "backends/cuda/fusion/tile.h"
#include "backends/cuda/state.h"
#include "core/runtime/fusion/key.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        std::uint64_t combine(std::uint64_t hash, std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
            return hash;
        }

        struct Entry {
            Entry(
                int selected_device,
                int selected_major,
                int selected_minor,
                const core::runtime::fusion::Expr& expression,
                std::shared_ptr<Compiled> compiled_kernel
            )
                : device(selected_device),
                  major(selected_major),
                  minor(selected_minor),
                  key(expression),
                  compiled(std::move(compiled_kernel)) {}

            int device;
            int major;
            int minor;
            core::runtime::fusion::Key key;
            std::shared_ptr<Compiled> compiled;

            bool matches(
                int requested_device,
                int requested_major,
                int requested_minor,
                const core::runtime::fusion::Expr& expression
            ) const {
                return
                    device == requested_device &&
                    major == requested_major &&
                    minor == requested_minor &&
                    key.matches(expression);
            }
        };

        using Cache = std::unordered_map<std::uint64_t, std::vector<Entry>>;

        // Reuse the caller's hash storage and preserve the inlined lookup path.
        __forceinline__ const Entry* find_entry(
            const Cache& cache,
            const std::uint64_t& hash,
            int device,
            int major,
            int minor,
            const core::runtime::fusion::Expr& expression
        ) {
            const auto found = cache.find(hash);
            if (found == cache.end()) {
                return nullptr;
            }
            for (const auto& entry : found->second) {
                if (entry.matches(device, major, minor, expression)) {
                    return &entry;
                }
            }
            return nullptr;
        }

        std::shared_ptr<Compiled> build(
            const core::runtime::fusion::Expr& expression,
            const State& cuda_state,
            int major,
            int minor
        ) {
            auto tiled = tile(expression);
            const auto source = emit(expression);
            const auto artifact = cache::get(
                source,
                nvrtc::context(major, minor),
                [&] {
                    return nvrtc::compile(source, major, minor);
                }
            );
            return load(artifact.ptx, expression, cuda_state, std::move(tiled));
        }
    }

    std::shared_ptr<Compiled> compile(
        const core::runtime::fusion::Expr& expression,
        const State& cuda_state
    ) {
        static std::mutex mutex;
        static std::atomic<std::shared_ptr<const Cache>> published{
            std::make_shared<const Cache>()
        };

        const auto device = cuda_state.index();
        const auto major = cuda_state.compute_major();
        const auto minor = cuda_state.compute_minor();
        auto hash = core::runtime::fusion::Key::hash(expression);
        hash = combine(hash, static_cast<std::uint64_t>(device));
        hash = combine(hash, static_cast<std::uint64_t>(major));
        hash = combine(hash, static_cast<std::uint64_t>(minor));

        auto current = published.load(std::memory_order_acquire);
        if (const auto* entry = find_entry(*current, hash, device, major, minor, expression)) {
            return entry->compiled;
        }

        std::lock_guard lock(mutex);
        current = published.load(std::memory_order_acquire);
        if (const auto* entry = find_entry(*current, hash, device, major, minor, expression)) {
            return entry->compiled;
        }

        auto compiled = build(expression, cuda_state, major, minor);
        auto next = std::make_shared<Cache>(*current);
        (*next)[hash].push_back(Entry{
            device,
            major,
            minor,
            expression,
            compiled
        });
        published.store(std::move(next), std::memory_order_release);
        return compiled;
    }
}
