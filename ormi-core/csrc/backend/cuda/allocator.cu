#include <cuda.h>
#include <unordered_map>
#include <vector>
#include <mutex>

#include "core/allocator.h"
#include "utils/macros.h"
#include "core/dtype.h"

namespace ormi {
    class CUDAAllocator : public IAllocator {
    private:
        CUdevice cuDevice;
        CUcontext cuContext;
        
        std::mutex pool_mutex;
        std::unordered_map<size_t, std::vector<void*>> free_blocks;

    public:
        CUDAAllocator() {
            CUDA_SAFE_CALL(cuInit(0));
            CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
            CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
            CUDA_SAFE_CALL(cuCtxSetCurrent(cuContext));
        }

        ~CUDAAllocator() override {
            for (auto& pair : free_blocks) {
                for (void* ptr : pair.second) {
                    cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
                }
            }
            cuDevicePrimaryCtxRelease(cuDevice);
        }

        void* allocate(size_t bytes) override {
            std::lock_guard<std::mutex> lock(pool_mutex);
            
            auto it = free_blocks.find(bytes);
            if (it != free_blocks.end() && !it->second.empty()) {
                void* ptr = it->second.back();
                it->second.pop_back();
                return ptr;
            }
            
            CUdeviceptr d_ptr;
            CUDA_SAFE_CALL(cuMemAlloc(&d_ptr, bytes));
            return reinterpret_cast<void*>(d_ptr);
        }

        void free(void* ptr, size_t bytes) override {
            if (!ptr) return;
            std::lock_guard<std::mutex> lock(pool_mutex);
            free_blocks[bytes].push_back(ptr);
        }

        void memset(void* ptr, int value, size_t bytes) override {
            CUDA_SAFE_CALL(cuMemsetD8(reinterpret_cast<CUdeviceptr>(ptr), value, bytes));
        }

        void memcpy(void* dst, const void* src, size_t size) override {
            CUDA_SAFE_CALL(cuMemcpyDtoD(
                reinterpret_cast<CUdeviceptr>(dst),
                reinterpret_cast<CUdeviceptr>(const_cast<void*>(src)),
                size
            ));
        }

        void fill_n(void* dst, const void* val_ptr, size_t elements, DType dtype) override {
            CUdeviceptr dst_ptr = reinterpret_cast<CUdeviceptr>(dst);

            switch (dtype) {
                case DType::Float32:
                case DType::Int32: {
                    unsigned int val = *static_cast<const unsigned int*>(val_ptr);
                    CUDA_SAFE_CALL(cuMemsetD32(dst_ptr, val, elements));
                    break;
                }
                case DType::Bool: {
                    unsigned char val = *static_cast<const unsigned char*>(val_ptr);
                    CUDA_SAFE_CALL(cuMemsetD8(dst_ptr, val, elements));
                    break;
                }
                default:
                    throw std::runtime_error("DType is not supported.");
            }
        }

        void copy_device_to_host(void* dst, const void* src, size_t bytes) override {
            CUDA_SAFE_CALL(cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(const_cast<void*>(src)), bytes));
        }

        void copy_host_to_device(void* dst, const void* src, size_t bytes) override {
            CUDA_SAFE_CALL(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, bytes));
        }

        bool is_host_visible() const override {
            return false;
        }
    };

    IAllocator* get_cuda_allocator() {
        static CUDAAllocator instance;
        return &instance;
    }
}