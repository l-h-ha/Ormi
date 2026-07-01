#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <mutex>
#include <limits>
#include <string>

#if defined(_MSC_VER)
    #include <malloc.h>
#endif

#include "core/allocator.h"
#include "core/dtype.h"
#include "core/error.h"
#include "core/float16.h"

namespace ormi {
    namespace {
        constexpr size_t kAlignment = 64;

        inline bool is_power_of_two(size_t x) {
            return x != 0 && (x & (x - 1)) == 0;
        }

        inline size_t align_size(size_t bytes, size_t alignment = kAlignment) {
            if (bytes == 0) return 0;

            if (!is_power_of_two(alignment)) {
                ORMI_THROW(MemoryError, "Alignment must be a power of two.");
            }

            if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
                ORMI_THROW(MemoryError, "Allocation size overflow.");
            }

            return (bytes + alignment - 1) & ~(alignment - 1);
        }

        inline size_t checked_element_count(size_t bytes, DType dtype) {
            const size_t item_size = dtype_size(dtype);

            if (bytes % item_size != 0) {
                ORMI_THROW(
                    DTypeError,
                    "fill_n byte count is not divisible by dtype size."
                );
            }

            return bytes / item_size;
        }

        inline void* aligned_malloc(size_t bytes, size_t alignment) {
            if (bytes == 0) return nullptr;

        #if defined(_MSC_VER)
            return _aligned_malloc(bytes, alignment);
        #else
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment, bytes) != 0) {
                return nullptr;
            }
            return ptr;
        #endif
        }

        inline void aligned_free(void* ptr) {
            if (!ptr) return;

        #if defined(_MSC_VER)
            _aligned_free(ptr);
        #else
            std::free(ptr);
        #endif
        }

    }

    class CPUAllocator final : public IAllocator {
    private:
        struct AllocationInfo {
            size_t bytes = 0;
        };

        std::mutex mutex_;
        std::unordered_map<size_t, std::vector<void*>> free_blocks_;
        std::unordered_map<void*, AllocationInfo> live_blocks_;

    private:
        void release_cache_locked() {
            for (auto& pair : free_blocks_) {
                for (void* ptr : pair.second) {
                    aligned_free(ptr);
                }
            }

            free_blocks_.clear();
        }

    public:
        ~CPUAllocator() override {
            std::lock_guard<std::mutex> lock(mutex_);

            release_cache_locked();

            // Best-effort cleanup of leaked live allocations.
            // Ideally this map should be empty at shutdown.
            for (auto& kv : live_blocks_) {
                aligned_free(kv.first);
            }

            live_blocks_.clear();
        }

        void* allocate(size_t bytes) override {
            if (bytes == 0) return nullptr;

            const size_t aligned_bytes = align_size(bytes);

            {
                std::lock_guard<std::mutex> lock(mutex_);

                auto it = free_blocks_.find(aligned_bytes);
                if (it != free_blocks_.end() && !it->second.empty()) {
                    void* ptr = it->second.back();
                    it->second.pop_back();

                    live_blocks_.emplace(ptr, AllocationInfo{aligned_bytes});
                    return ptr;
                }
            }

            void* ptr = aligned_malloc(aligned_bytes, kAlignment);

            if (!ptr) {
                // release cached memory and retry.
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    release_cache_locked();
                }

                ptr = aligned_malloc(aligned_bytes, kAlignment);

                if (!ptr) {
                    ORMI_THROW(
                        MemoryError,
                        "CPU Out of Memory: Failed to allocate " +
                        std::to_string(aligned_bytes) + " bytes."
                    );
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);

                auto inserted = live_blocks_.emplace(
                    ptr,
                    AllocationInfo{aligned_bytes}
                );

                if (!inserted.second) {
                    aligned_free(ptr);

                    ORMI_THROW(
                        MemoryError,
                        "CPUAllocator internal error: duplicate allocation pointer."
                    );
                }
            }

            return ptr;
        }

        void free(void* ptr, size_t /*bytes*/) override {
            if (!ptr) return;

            size_t actual_bytes = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);

                auto it = live_blocks_.find(ptr);
                if (it == live_blocks_.end()) {
                    ORMI_THROW(
                        MemoryError,
                        "CPUAllocator double free or attempt to free unknown pointer."
                    );
                }

                actual_bytes = it->second.bytes;
                live_blocks_.erase(it);

                free_blocks_[actual_bytes].push_back(ptr);
            }
        }

        void memset(void* ptr, int value, size_t bytes) override {
            if (!ptr || bytes == 0) return;
            std::memset(ptr, value, bytes);
        }

        void memcpy(void* dst, const void* src, size_t size) override {
            if (!dst || !src || size == 0) return;
            std::memcpy(dst, src, size);
        }

        void fill_n(void* dst, const void* val_ptr, size_t elements, DType dtype) override {
            if (!dst || !val_ptr || elements == 0) return;

            switch (dtype) {
                case DType::Float32: {
                    float val;
                    std::memcpy(&val, val_ptr, sizeof(float));
                    std::fill_n(static_cast<float*>(dst), elements, val);
                    break;
                }
                case DType::Float16: {
                    float val;
                    std::memcpy(&val, val_ptr, sizeof(float));

                    uint16_t fp16_val = fp16::float_to_bits(val);
                    std::fill_n(static_cast<uint16_t*>(dst), elements, fp16_val);
                    break;
                }
                case DType::Int32: {
                    int32_t val;
                    std::memcpy(&val, val_ptr, sizeof(int32_t));
                    std::fill_n(static_cast<int32_t*>(dst), elements, val);
                    break;
                }
                case DType::Bool: {
                    bool val;
                    std::memcpy(&val, val_ptr, sizeof(bool));

                    // Store Bool as 1 byte per element.
                    std::memset(dst, val ? 1 : 0, elements);
                    break;
                }
                default:
                    ORMI_THROW(
                        DTypeError,
                        "Unrecognized DType (" + to_string(dtype) + ")"
                    );
            }
        }

        void copy_device_to_host(void* dst, const void* src, size_t bytes) override {
            if (!dst || !src || bytes == 0) return;
            std::memcpy(dst, src, bytes);
        }

        void copy_host_to_device(void* dst, const void* src, size_t bytes) override {
            if (!dst || !src || bytes == 0) return;
            std::memcpy(dst, src, bytes);
        }

        bool is_host_visible() const override {
            return true;
        }

        void empty_cache() override {
            std::lock_guard<std::mutex> lock(mutex_);
            release_cache_locked();
        }
    };

    IAllocator* get_cpu_allocator() {
        static CPUAllocator instance;
        return &instance;
    }
}
