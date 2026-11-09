#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/reduction/index.cuh"
#include "backends/cuda/reduction/max.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        template <typename T, typename Value>
        __global__ void maximum_kernel(
            T* out,
            const T* input,
            std::size_t outputs,
            std::size_t count,
            reduction::Index index,
            bool contiguous
        ) {
            __shared__ Value partial[256];
            for (std::size_t output = blockIdx.x; output < outputs; output += gridDim.x) {
                const auto base = contiguous ? output * count : reduction::input_base_offset(output, index);
                Value value = -static_cast<Value>(CUDART_INF);
                for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
                    const auto offset = base + (contiguous ? i : reduction::reduced_offset(i, index));
                    Value candidate;
                    if constexpr (std::is_same_v<T, __half>) {
                        candidate = __half2float(input[offset]);
                    } else {
                        candidate = input[offset];
                    }
                    if (candidate > value || isnan(candidate)) {
                        value = candidate;
                    }
                }
                partial[threadIdx.x] = value;
                __syncthreads();
                for (unsigned width = blockDim.x / 2; width > 0; width /= 2) {
                    if (threadIdx.x < width) {
                        const auto candidate = partial[threadIdx.x + width];
                        if (candidate > partial[threadIdx.x] || isnan(candidate)) {
                            partial[threadIdx.x] = candidate;
                        }
                    }
                    __syncthreads();
                }
                if (threadIdx.x == 0) {
                    if constexpr (std::is_same_v<T, __half>) {
                        out[output] = __float2half_rn(partial[0]);
                    } else {
                        out[output] = partial[0];
                    }
                }
                __syncthreads();
            }
        }

        template <typename T, typename Value>
        void launch(
            core::Tensor& out,
            const core::Tensor& input,
            std::span<const std::int64_t> axes,
            cudaStream_t stream
        ) {
            const auto index = reduction::make_index(input, axes);
            std::size_t count = 1;
            for (const auto axis : axes) {
                count *= static_cast<std::size_t>(input.shape()[static_cast<std::size_t>(axis)]);
            }
            bool contiguous = input.is_contiguous();
            for (std::size_t i = 0; i < axes.size(); ++i) {
                contiguous =
                    contiguous && static_cast<std::size_t>(axes[i]) == input.rank() - axes.size() + i;
            }
            const auto blocks = static_cast<unsigned>(out.numel() < 65535 ? out.numel() : 65535);
            maximum_kernel<T, Value><<<blocks, 256, 0, stream>>>(
                static_cast<T*>(out.data()),
                static_cast<const T*>(input.data()),
                out.numel(),
                count,
                index,
                contiguous
            );
            check(cudaGetLastError(), "CUDA max reduction");
        }
    }

    void reduce_max(core::Tensor& out, const core::Tensor& input, std::span<const std::int64_t> axes) {
        if (out.numel() == 0) {
            return;
        }
        auto& device = state(out.device());
        check(cudaSetDevice(device.index()), "cudaSetDevice");
        switch (out.dtype()) {
            case core::DType::float64:
                launch<double, double>(out, input, axes, device.stream());
                return;
            case core::DType::float32:
                launch<float, float>(out, input, axes, device.stream());
                return;
            case core::DType::float16:
                launch<__half, float>(out, input, axes, device.stream());
                return;
            default:
                ORMI_THROW(DTypeError, "CUDA max requires floating-point tensors.");
        }
    }
}
