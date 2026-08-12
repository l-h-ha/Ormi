#include <cstddef>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include <ormi/core/device.h>
#include <ormi/core/error.h>

#include "backends/cpu/allocator.h"

namespace ormi::backends::cpu {
    namespace {
        constexpr std::size_t alignment = 64;
        constexpr std::size_t cache_limit = 64 * 1024 * 1024;

        bool is_power_of_two(std::size_t value) {
            return value != 0 && (value & (value - 1)) == 0;
        }

        std::size_t align_size(std::size_t bytes) {
            if (bytes == 0) {
                return 0;
            }

            if (!is_power_of_two(alignment)) {
                ORMI_THROW(MemoryError, "CPU allocator alignment must be a power of two.");
            }

            if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
                ORMI_THROW(MemoryError, "CPU allocation size overflow.");
            }

            return (bytes + alignment - 1) & ~(alignment - 1);
        }

        void* aligned_allocate(std::size_t bytes) {
            if (bytes == 0) {
                return nullptr;
            }
            return ::operator new(
                bytes,
                std::align_val_t{alignment},
                std::nothrow
            );
        }

        void aligned_deallocate(void* ptr) {
            if (!ptr) {
                return;
            }

            ::operator delete(ptr, std::align_val_t{alignment});
        }

        class cpuAllocator final : public core::memory::Allocator {
        public:
            ~cpuAllocator() override {
                empty_cache();
            }

            void* allocate(std::size_t bytes) override {
                const std::size_t aligned_bytes = align_size(bytes);

                if (aligned_bytes == 0) {
                    return nullptr;
                }

                {
                    std::lock_guard<std::mutex> lock(mutex_);

                    auto it = free_blocks_.find(aligned_bytes);
                    if (it != free_blocks_.end() && !it->second.empty()) {
                        void* ptr = it->second.back();
                        it->second.pop_back();
                        cached_bytes_ -= aligned_bytes;
                        return ptr;
                    }
                }

                void* ptr = aligned_allocate(aligned_bytes);

                if (!ptr) {
                    ORMI_THROW(
                        MemoryError,
                        "CPU allocation failed for " + std::to_string(aligned_bytes) + " bytes."
                    );
                }

                return ptr;
            }

            void deallocate(void* ptr, std::size_t bytes) noexcept override {
                if (!ptr) {
                    return;
                }

                const std::size_t aligned_bytes = align_size(bytes);

                if (aligned_bytes == 0) {
                    aligned_deallocate(ptr);
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(mutex_);

                    if (cached_bytes_ + aligned_bytes <= cache_limit) {
                        free_blocks_[aligned_bytes].push_back(ptr);
                        cached_bytes_ += aligned_bytes;
                        return;
                    }
                }

                aligned_deallocate(ptr);
            }

            void empty_cache() override {
                std::unordered_map<std::size_t, std::vector<void*>> blocks;

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    blocks.swap(free_blocks_);
                    cached_bytes_ = 0;
                }

                for (auto& [_, pointers] : blocks) {
                    for (void* ptr : pointers) {
                        aligned_deallocate(ptr);
                    }
                }
            }

            bool host_visible() const override {
                return true;
            }

        private:
            std::mutex mutex_;
            std::unordered_map<std::size_t, std::vector<void*>> free_blocks_;
            std::size_t cached_bytes_ = 0;
        };
    }

    core::memory::Allocator* allocator() {
        static cpuAllocator allocator;
        return &allocator;
    }

    core::memory::Allocator* get_allocator([[maybe_unused]] core::Device device) {
        return allocator();
    }
}
