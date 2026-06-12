#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <mutex>

#include "core/allocator.h"
#include "core/dtype.h"

namespace ormi {
    class CPUAllocator : public IAllocator {
    private:
        std::mutex pool_mutex;
        std::unordered_map<size_t, std::vector<void*>> free_blocks;

    public:
        ~CPUAllocator() override {
            for (auto& pair : free_blocks) {
                for (void* ptr : pair.second) {
                    std::free(ptr);
                }
            }
        }

        void* allocate(size_t bytes) override {
            std::lock_guard<std::mutex> lock(pool_mutex);
            auto it = free_blocks.find(bytes);
            if (it != free_blocks.end() && !it->second.empty()) {
                void* ptr = it->second.back();
                it->second.pop_back();
                return ptr;
            }
            return std::malloc(bytes); 
        }

        void free(void* ptr, size_t bytes) override {
            std::lock_guard<std::mutex> lock(pool_mutex);
            free_blocks[bytes].push_back(ptr);
        }

        void memset(void* ptr, int value, size_t bytes) override {
            std::memset(ptr, value, bytes);
        }

        void memcpy(void* dst, const void* src, size_t size) override {
            std::memcpy(dst, src, size);
        }

        void fill_n(void* dst, const void* val_ptr, size_t elements, DType dtype) override {
            switch (dtype) {
                case DType::Float32: {
                    float val = *static_cast<const float*>(val_ptr);
                    std::fill_n(static_cast<float*>(dst), elements, val);
                    break;
                }
                case DType::Int32: {
                    int32_t val = *static_cast<const int32_t*>(val_ptr);
                    std::fill_n(static_cast<int32_t*>(dst), elements, val);
                    break;
                }
                case DType::Bool: {
                    bool val = *static_cast<const bool*>(val_ptr);
                    std::fill_n(static_cast<bool*>(dst), elements, val);
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported DType.");
            }
        }

        void copy_device_to_host(void* dst, const void* src, size_t bytes) override {
            std::memcpy(dst, src, bytes);
        }

        void copy_host_to_device(void* dst, const void* src, size_t bytes) override {
            std::memcpy(dst, src, bytes);
        }

        bool is_host_visible() const override {
            return true;
        }
    };

    IAllocator* get_cpu_allocator() {
        static CPUAllocator instance;
        return &instance;
    }
}