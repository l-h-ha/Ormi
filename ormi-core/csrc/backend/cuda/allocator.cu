#include <cuda.h>
#include <cuda_fp16.h>

#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <cstdlib>

#include "core/allocator.h"
#include "utils/macros.h"
#include "core/dtype.h"
#include "backend/cuda/utils.cuh"
#include "core/error.h"

namespace ormi {
    namespace {
        inline size_t align_size(size_t bytes, size_t alignment = 256) {
            if (bytes == 0) return 0;

            if (bytes > std::numeric_limits<size_t>::max() - alignment + 1) {
                ORMI_THROW(MemoryError, "Allocation size overflow.");
            }

            return (bytes + alignment - 1) & ~(alignment - 1);
        }

        inline uint16_t fp32_to_fp16_bits(float value) {
            __half h = __float2half_rn(value);

            uint16_t bits;
            std::memcpy(&bits, &h, sizeof(bits));

            return bits;
        }

        inline cuuint64_t parse_env_u64(const char* name, cuuint64_t fallback) {
            const char* raw = std::getenv(name);
            if (!raw || !*raw) return fallback;

            char* end = nullptr;
            unsigned long long value = std::strtoull(raw, &end, 10);

            if (end == raw) return fallback;
            return static_cast<cuuint64_t>(value);
        }
    }

    class CUDAAllocator final : public IAllocator {
    private:
        CUdevice cuDevice{};
        CUcontext cuContext{};

        int major = 0;
        int minor = 0;

        bool is_emulated_fp16 = false;
        bool async_pool_supported = false;

    #if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
        CUmemoryPool mempool{};
    #endif

        struct AllocationInfo {
            size_t bytes = 0;
        };

        mutable std::mutex mutex_;
        std::unordered_map<void*, AllocationInfo> live_blocks_;

        std::once_flag fp16_warn_once_;

    private:
        void set_context() const {
            CUDA_SAFE_CALL(cuCtxSetCurrent(cuContext));
        }

        void init_pool() {
        #if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
            int supported = 0;

            CUresult attr_status = cuDeviceGetAttribute(
                &supported,
                CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                cuDevice
            );

            if (attr_status != CUDA_SUCCESS || supported == 0) {
                async_pool_supported = false;
                ORMI_WARN(
                    "CUDA stream-ordered memory pools are unavailable. "
                    "Falling back to cuMemAlloc/cuMemFree."
                );
                return;
            }

            CUmemLocation location{};
            location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            location.id = static_cast<int>(cuDevice);

            CUDA_SAFE_CALL(cuMemGetDefaultMemPool(
                &mempool,
                &location,
                CU_MEM_ALLOCATION_TYPE_PINNED
            ));

            /*
            * CUDA's default memory-pool release threshold is usually conservative.
            *
            * Override with:
            *   ORMI_CUDA_MEMPOOL_RELEASE_THRESHOLD=<bytes>
            *
            * Examples:
            *   ORMI_CUDA_MEMPOOL_RELEASE_THRESHOLD=0
            *     Release memory aggressively. Lower VRAM footprint, slower repeated allocs.
            *
            *   ORMI_CUDA_MEMPOOL_RELEASE_THRESHOLD=8589934592
            *     Keep up to 8 GiB reserved in the CUDA pool.
            *
            *   ORMI_CUDA_MEMPOOL_RELEASE_THRESHOLD=18446744073709551615
            *     Keep memory as aggressively as possible.
            */
            cuuint64_t release_threshold = parse_env_u64(
                "ORMI_CUDA_MEMPOOL_RELEASE_THRESHOLD",
                8ull * 1024ull * 1024ull * 1024ull
            );

            CUDA_SAFE_CALL(cuMemPoolSetAttribute(
                mempool,
                CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &release_threshold
            ));

            async_pool_supported = true;
        #else
            async_pool_supported = false;
            ORMI_WARN(
                "CUDA headers are older than CUDA 11.2. "
                "Falling back to cuMemAlloc/cuMemFree."
            );
        #endif
        }

    public:
        CUDAAllocator() {
            CUDA_SAFE_CALL(cuInit(0));
            CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
            CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
            set_context();

            CUDA_SAFE_CALL(cuDeviceGetAttribute(
                &major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cuDevice
            ));

            CUDA_SAFE_CALL(cuDeviceGetAttribute(
                &minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cuDevice
            ));

            if (major < 7) {
                is_emulated_fp16 = true;
                ORMI_WARN(
                    "Detected older CUDA GPU. FP16 throughput may be poor. "
                    "Prefer Float32 unless memory bandwidth/capacity is the bottleneck."
                );
            } else if (major == 7 && minor == 5) {
                char name[256];
                CUDA_SAFE_CALL(cuDeviceGetName(name, sizeof(name), cuDevice));

                std::string device_name(name);

                if (device_name.find("GTX") != std::string::npos ||
                    device_name.find("1650") != std::string::npos ||
                    device_name.find("1660") != std::string::npos) {
                    is_emulated_fp16 = true;
                }
            }

            init_pool();
        }

        ~CUDAAllocator() override {
            cuCtxSetCurrent(cuContext);

            {
                std::lock_guard<std::mutex> lock(mutex_);

                if (!live_blocks_.empty()) {
                    ORMI_WARN(
                        "CUDAAllocator destroyed with " +
                        std::to_string(live_blocks_.size()) +
                        " live allocation(s). This indicates leaked Tensor/Buffer ownership."
                    );
                }

                // Synchronize first. This avoids freeing memory
                // while kernels may still be using it.
                cuCtxSynchronize();

                for (auto& kv : live_blocks_) {
                    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(kv.first);

                #if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
                    if (async_pool_supported) {
                        CUstream stream = cuda::get_stream();
                        cuMemFreeAsync(dptr, stream);
                    } else
                #endif
                    {
                        cuMemFree(dptr);
                    }
                }

            #if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
                if (async_pool_supported) {
                    CUstream stream = cuda::get_stream();
                    cuStreamSynchronize(stream);
                    cuMemPoolTrimTo(mempool, 0);
                }
            #endif

                live_blocks_.clear();
            }

            cuDevicePrimaryCtxRelease(cuDevice);
        }

        void* allocate(size_t bytes) override {
            if (bytes == 0) return nullptr;

            set_context();

            const size_t aligned_bytes = align_size(bytes);
            CUdeviceptr dptr = 0;

            CUresult status = CUDA_SUCCESS;

        #if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
            if (async_pool_supported) {
                CUstream stream = cuda::get_stream();

                status = cuMemAllocFromPoolAsync(
                    &dptr,
                    aligned_bytes,
                    mempool,
                    stream
                );

                if (status == CUDA_ERROR_OUT_OF_MEMORY) {
                    // Wait for pending async frees, trim unused pool mem, retry.
                    cuStreamSynchronize(stream);
                    cuMemPoolTrimTo(mempool, 0);

                    status = cuMemAllocFromPoolAsync(
                        &dptr,
                        aligned_bytes,
                        mempool,
                        stream
                    );
                }
            } else
        #endif
            {
                status = cuMemAlloc(&dptr, aligned_bytes);
            }

            if (status == CUDA_ERROR_OUT_OF_MEMORY) {
                ORMI_THROW(
                    MemoryError,
                    "CUDA Out of Memory: failed to allocate " +
                    std::to_string(aligned_bytes) + " bytes."
                );
            }

            if (status != CUDA_SUCCESS) {
                ORMI_THROW(
                    MemoryError,
                    "CUDA allocation failed with error code " + std::to_string(status)
                );
            }

            void* ptr = reinterpret_cast<void*>(dptr);

            {
                std::lock_guard<std::mutex> lock(mutex_);

                auto inserted = live_blocks_.emplace(
                    ptr,
                    AllocationInfo{aligned_bytes}
                );

                if (!inserted.second) {
                    // Unlikely, but protects allocator bookkeeping.
                #if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
                    if (async_pool_supported) {
                        cuMemFreeAsync(dptr, cuda::get_stream());
                    } else
                #endif
                    {
                        cuMemFree(dptr);
                    }

                    ORMI_THROW(
                        MemoryError,
                        "CUDAAllocator internal error: duplicate allocation pointer."
                    );
                }
            }

            return ptr;
        }

        void free(void* ptr, size_t /*bytes*/) override {
            if (!ptr) return;

            set_context();

            {
                std::lock_guard<std::mutex> lock(mutex_);

                auto it = live_blocks_.find(ptr);
                if (it == live_blocks_.end()) {
                    ORMI_THROW(
                        MemoryError,
                        "CUDAAllocator double free or attempt to free unknown pointer."
                    );
                }

                live_blocks_.erase(it);
            }

            CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);

        #if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
            if (async_pool_supported) {
                CUDA_SAFE_CALL(cuMemFreeAsync(dptr, cuda::get_stream()));
                return;
            }
        #endif

            CUDA_SAFE_CALL(cuMemFree(dptr));
        }

        void memset(void* ptr, int value, size_t bytes) override {
            if (!ptr || bytes == 0) return;

            set_context();

            CUDA_SAFE_CALL(cuMemsetD8Async(
                reinterpret_cast<CUdeviceptr>(ptr),
                static_cast<unsigned char>(value),
                bytes,
                cuda::get_stream()
            ));
        }

        void memcpy(void* dst, const void* src, size_t size) override {
            if (!dst || !src || size == 0) return;

            set_context();
            CUDA_SAFE_CALL(cuMemcpyDtoDAsync(
                reinterpret_cast<CUdeviceptr>(dst),
                reinterpret_cast<CUdeviceptr>(const_cast<void*>(src)),
                size,
                cuda::get_stream()
            ));
        }

        void fill_n(void* dst, const void* val_ptr, size_t elements, DType dtype) override {
            if (!dst || !val_ptr || elements == 0) return;

            set_context();

            CUdeviceptr dst_ptr = reinterpret_cast<CUdeviceptr>(dst);
            CUstream stream = cuda::get_stream();

            switch (dtype) {
                case DType::Float32:
                case DType::Int32: {
                    uint32_t bits;
                    std::memcpy(&bits, val_ptr, sizeof(uint32_t));

                    CUDA_SAFE_CALL(cuMemsetD32Async(
                        dst_ptr,
                        bits,
                        elements,
                        stream
                    ));

                    break;
                }
                case DType::Float16: {
                    if (is_emulated_fp16) {
                        std::call_once(fp16_warn_once_, [] {
                            ORMI_WARN(
                                "Using Float16 on hardware with poor FP16 throughput. "
                                "Prefer Float32 unless memory bandwidth/capacity is the bottleneck."
                            );
                        });
                    }

                    float value;
                    std::memcpy(&value, val_ptr, sizeof(float));

                    uint16_t bits = fp32_to_fp16_bits(value);
                    CUDA_SAFE_CALL(cuMemsetD16Async(
                        dst_ptr,
                        bits,
                        elements,
                        stream
                    ));

                    break;
                }
                case DType::Bool: {
                    bool value;
                    std::memcpy(&value, val_ptr, sizeof(bool));

                    CUDA_SAFE_CALL(cuMemsetD8Async(
                        dst_ptr,
                        value ? 1 : 0,
                        elements,
                        stream
                    ));

                    break;
                }

                default:
                    ORMI_THROW(DTypeError, "Unrecognized DType. (CUDA)");
            }
        }

        void copy_device_to_host(void* dst, const void* src, size_t bytes) override {
            if (!dst || !src || bytes == 0) return;
            set_context();

            CUstream stream = cuda::get_stream();
            CUDA_SAFE_CALL(cuMemcpyDtoHAsync(
                dst,
                reinterpret_cast<CUdeviceptr>(const_cast<void*>(src)),
                bytes,
                stream
            ));

            CUDA_SAFE_CALL(cuStreamSynchronize(stream));
        }

        void copy_host_to_device(void* dst, const void* src, size_t bytes) override {
            if (!dst || !src || bytes == 0) return;
            set_context();

            CUDA_SAFE_CALL(cuMemcpyHtoDAsync(
                reinterpret_cast<CUdeviceptr>(dst),
                src,
                bytes,
                cuda::get_stream()
            ));
        }

        bool is_host_visible() const override {
            return false;
        }

        void empty_cache() override {
            set_context();

        #if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
            if (async_pool_supported) {
                // cuMemPoolTrimTo cannot release memory still considered outstanding.
                // Async frees may still be pending until synchronized.
                CUDA_SAFE_CALL(cuCtxSynchronize());
                CUDA_SAFE_CALL(cuMemPoolTrimTo(mempool, 0));
            }
        #endif
        }
    };

    IAllocator* get_cuda_allocator() {
        static CUDAAllocator instance;
        return &instance;
    }
}