#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>

#include "backends/cuda/allocator.h"
#include "backends/cuda/check.cuh"
#include "backends/cuda/state.h"
#include "backends/cuda/utils.cuh"

namespace ormi::backends::cuda {
    namespace {
        class cudaAllocator final : public core::memory::Allocator {
        public:
            explicit cudaAllocator(State& state)
                : state_(state) {}

            void* allocate(std::size_t bytes) override {
                if (bytes == 0) {
                    return nullptr;
                }

                check(cudaSetDevice(state_.index()), "cudaSetDevice");

                void* ptr = nullptr;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
                if (state_.has_pool()) {
                    cudaError_t status = cudaMallocFromPoolAsync(&ptr, bytes, state_.pool(), state_.stream());

                    if (status == cudaErrorMemoryAllocation) {
                        check(cudaStreamSynchronize(state_.stream()), "cudaStreamSynchronize");
                        check(cudaMemPoolTrimTo(state_.pool(), 0), "cudaMemPoolTrimTo");

                        status = cudaMallocFromPoolAsync(&ptr, bytes, state_.pool(), state_.stream());
                    }

                    check(status, "cudaMallocFromPoolAsync");
                    return ptr;
                }
#endif

                check(cudaMalloc(&ptr, bytes), "cudaMalloc");
                return ptr;
            }

            void deallocate(void* ptr, [[maybe_unused]] std::size_t bytes) noexcept override {
                if (!ptr) {
                    return;
                }

                if (cudaSetDevice(state_.index()) != cudaSuccess) {
                    return;
                }

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
                if (state_.has_pool()) {
                    (void)cudaFreeAsync(ptr, state_.stream());
                    return;
                }
#endif

                (void)cudaFree(ptr);
            }

            void empty_cache() override {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
                if (!state_.has_pool()) {
                    return;
                }

                check(cudaSetDevice(state_.index()), "cudaSetDevice");
                check(cudaStreamSynchronize(state_.stream()), "cudaStreamSynchronize");
                check(cudaMemPoolTrimTo(state_.pool(), 0), "cudaMemPoolTrimTo");
#endif
            }

            bool host_visible() const override {
                return false;
            }

        private:
            State& state_;
        };
    }

    core::memory::Allocator* allocator(core::Device device) {
        static const int device_count = available_devices();
        static std::mutex mutex;
        static std::vector<std::unique_ptr<cudaAllocator>> allocators(static_cast<std::size_t>(device_count));

        const int index = device.index();
        check_device_id(index);

        std::lock_guard lock(mutex);

        auto& slot = allocators[static_cast<std::size_t>(index)];

        if (!slot) {
            slot = std::make_unique<cudaAllocator>(state(device));
        }

        return slot.get();
    }
}
