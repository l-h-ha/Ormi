#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "backend/cuda/utils.cuh"
#include "core/dtype.h"
#include "core/error.h"
#include "core/buffer.h"
#include "optimizer.cuh"
#include "utils/macros.h"
#include "backend/cuda/config.cuh"

namespace ormi::cuda {
    namespace {
        __global__ void sgd_kernel_fp32(
            float* __restrict__ w,
            const float* __restrict__ g,
            float lr,
            size_t n
        ) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            if (i < n) {
                w[i] -= g[i] * lr;
            }
        }

        __global__ void sgd_kernel_fp16(
            __half* __restrict__ w,
            const __half* __restrict__ g,
            __half lr,
            size_t n
        ) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            if (i < n) {
                w[i] = __hsub(w[i], __hmul(g[i], lr));
            }
        }
    }

    void sgd_step(
        const std::shared_ptr<Buffer>& grad,
        const std::shared_ptr<Buffer>& weight,
        float lr
    ) {
        if (!grad || !weight || !grad->ptr || !weight->ptr) {
            return;
        }

        if (lr == 0.0f) {
            return;
        }

        if (grad->bytes != weight->bytes) {
            ORMI_THROW(
                ShapeError,
                "CUDA SGD requires grad and weight to have the same byte size."
            );
        }

        if (grad->dtype != weight->dtype) {
            ORMI_THROW(
                DTypeError,
                "CUDA SGD requires grad and weight to have the same dtype."
            );
        }

        const size_t dtype_bytes = dtype_size(weight->dtype);
        const size_t n = weight->bytes / dtype_bytes;

        if (n == 0) {
            return;
        }

        const int threads = get_optimizer_threads();
        const int blocks = static_cast<int>((n + threads - 1) / threads);

        cudaStream_t stream = cuda::get_stream();

        if (weight->dtype == DType::Float32) {
            auto* w_ptr = static_cast<float*>(weight->ptr);
            const auto* g_ptr = static_cast<const float*>(grad->ptr);

            sgd_kernel_fp32<<<blocks, threads, 0, stream>>>(
                w_ptr,
                g_ptr,
                lr,
                n
            );

            CUDART_SAFE_CALL(cudaPeekAtLastError());
        }
        else if (weight->dtype == DType::Float16) {
            auto* w_ptr = static_cast<__half*>(weight->ptr);
            const auto* g_ptr = static_cast<const __half*>(grad->ptr);

            const __half lr_half = __float2half(lr);

            sgd_kernel_fp16<<<blocks, threads, 0, stream>>>(
                w_ptr,
                g_ptr,
                lr_half,
                n
            );

            CUDART_SAFE_CALL(cudaPeekAtLastError());
        }
        else {
            ORMI_THROW(DTypeError, "Unsupported data type for SGD. (CUDA)");
        }
    }
}