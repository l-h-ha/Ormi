#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>

namespace ormi::backends::cuda {
    class State final {
    public:
        explicit State(int index);
        ~State() noexcept;

        State(const State&) = delete;
        State& operator=(const State&) = delete;

        int index() const noexcept;
        int compute_major() const noexcept;
        int compute_minor() const noexcept;
        int multiprocessors() const noexcept;
        cudaStream_t stream() const noexcept;
        cublasHandle_t cublas() const noexcept;
        bool has_pool() const noexcept;

    #if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        cudaMemPool_t pool() const noexcept;
    #endif

    private:
        int index_;
        int compute_major_ = 0;
        int compute_minor_ = 0;
        int multiprocessors_ = 0;
        cudaStream_t stream_ = nullptr;
        cublasHandle_t cublas_ = nullptr;
        bool has_pool_ = false;

    #if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        cudaMemPool_t pool_ = nullptr;
    #endif
    };

    State& state(core::Device device);
}
