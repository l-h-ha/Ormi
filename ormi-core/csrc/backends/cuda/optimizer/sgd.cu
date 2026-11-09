#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/elementwise/index.cuh"
#include "backends/cuda/optimizer/sgd.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        template <typename T>
        __device__ T updated(T parameter, T gradient, double step_size) {
            if constexpr (std::is_same_v<T, __half>) {
                return __float2half_rn(
                    __half2float(parameter) - static_cast<float>(step_size) * __half2float(gradient)
                );
            } else {
                return parameter - static_cast<T>(step_size) * gradient;
            }
        }

        template <typename T>
        __global__ void sgd_contiguous_kernel(
            T* out,
            const T* parameter,
            const T* gradient,
            std::size_t elements,
            double step_size
        ) {
            std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                out[linear] = updated(parameter[linear], gradient[linear], step_size);
            }
        }

        template <typename T>
        __global__ void sgd_indexed_kernel(
            T* out,
            const T* parameter,
            const T* gradient,
            std::size_t elements,
            double step_size,
            elementwise::BinaryIndex index
        ) {
            std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                std::size_t parameter_offset = 0;
                std::size_t gradient_offset = 0;
                elementwise::storage_offsets(linear, index, parameter_offset, gradient_offset);
                out[linear] = updated(parameter[parameter_offset], gradient[gradient_offset], step_size);
            }
        }

        template <typename T>
        void launch_sgd(
            core::Tensor& out,
            const core::Tensor& parameter,
            const core::Tensor& gradient,
            double step_size,
            cudaStream_t stream
        ) {
            const auto blocks = elementwise::blocks(out.numel());
            auto* out_data = static_cast<T*>(out.data());
            const auto* parameter_data = static_cast<const T*>(parameter.data());
            const auto* gradient_data = static_cast<const T*>(gradient.data());

            if (parameter.is_contiguous() && gradient.is_contiguous()) {
                sgd_contiguous_kernel<<<blocks, elementwise::threads, 0, stream>>>(
                    out_data,
                    parameter_data,
                    gradient_data,
                    out.numel(),
                    step_size
                );
            } else {
                const auto index = elementwise::make_index(out, parameter, gradient);
                sgd_indexed_kernel<<<blocks, elementwise::threads, 0, stream>>>(
                    out_data,
                    parameter_data,
                    gradient_data,
                    out.numel(),
                    step_size,
                    index
                );
            }

            check(cudaGetLastError(), "CUDA SGD kernel");
        }
    }

    void sgd(
        core::Tensor& out,
        const core::Tensor& parameter,
        const core::Tensor& gradient,
        double step_size
    ) {
        if (out.numel() == 0) {
            return;
        }

        auto& cuda_state = state(out.device());
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

        switch (out.dtype()) {
            case core::DType::float64:
                launch_sgd<double>(out, parameter, gradient, step_size, cuda_state.stream());
                return;
            case core::DType::float32:
                launch_sgd<float>(out, parameter, gradient, step_size, cuda_state.stream());
                return;
            case core::DType::float16:
                launch_sgd<__half>(out, parameter, gradient, step_size, cuda_state.stream());
                return;
            case core::DType::int32:
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CUDA SGD supports floating-point tensors only.");
        }

        ORMI_THROW(DTypeError, "CUDA SGD cannot handle an unknown dtype.");
    }
}
