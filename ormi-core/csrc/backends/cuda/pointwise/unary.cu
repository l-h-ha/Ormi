#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/elementwise/index.cuh"
#include "backends/cuda/pointwise.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        template <typename T, core::ops::UnaryOp Op>
        __device__ T apply_activation(T value) {
            if constexpr (Op == core::ops::UnaryOp::abs) {
                return ::fabs(value);
            } else if constexpr (Op == core::ops::UnaryOp::sign) {
                return value != value ? value : T((value > T(0)) - (value < T(0)));
            } else if constexpr (Op == core::ops::UnaryOp::positive) {
                return value != value ? value : T(value > T(0));
            } else if constexpr (Op == core::ops::UnaryOp::relu) {
                return value < T(0) ? T(0) : value;
            } else if constexpr (Op == core::ops::UnaryOp::tanh) {
                return ::tanh(value);
            } else if constexpr (Op == core::ops::UnaryOp::sigmoid) {
                return value >= T(0) ? T(1) / (T(1) + ::exp(-value)) : ::exp(value) / (T(1) + ::exp(value));
            } else if constexpr (Op == core::ops::UnaryOp::softplus) {
                return value > T(0) ? value + ::log1p(::exp(-value)) : ::log1p(::exp(value));
            } else if constexpr (Op == core::ops::UnaryOp::erf) {
                return ::erf(value);
            } else if constexpr (Op == core::ops::UnaryOp::log1p) {
                return ::log1p(value);
            } else if constexpr (Op == core::ops::UnaryOp::expm1) {
                return ::expm1(value);
            }
        }

        template <typename T, core::ops::UnaryOp Op>
        __device__ T apply_unary(T value) {
            if constexpr (Op == core::ops::UnaryOp::rsqrt) {
                if constexpr (std::is_same_v<T, __half>) {
                    return __float2half_rn(::rsqrtf(__half2float(value)));
                } else if constexpr (std::is_same_v<T, float>) {
                    return ::rsqrtf(value);
                } else {
                    return ::rsqrt(value);
                }
            } else if constexpr (Op == core::ops::UnaryOp::abs || Op == core::ops::UnaryOp::sign ||
                                 Op == core::ops::UnaryOp::positive || Op == core::ops::UnaryOp::tanh ||
                                 Op == core::ops::UnaryOp::sigmoid || Op == core::ops::UnaryOp::softplus ||
                                 Op == core::ops::UnaryOp::relu || Op == core::ops::UnaryOp::erf ||
                                 Op == core::ops::UnaryOp::log1p || Op == core::ops::UnaryOp::expm1) {
                if constexpr (std::is_same_v<T, __half>) {
                    return __float2half_rn(apply_activation<float, Op>(__half2float(value)));
                } else {
                    return apply_activation<T, Op>(value);
                }
            } else if constexpr (Op == core::ops::UnaryOp::neg) {
                if constexpr (std::is_same_v<T, __half>) {
                    return __hneg(value);
                } else {
                    return -value;
                }
            } else if constexpr (std::is_same_v<T, __half>) {
                const auto input = __half2float(value);

                if constexpr (Op == core::ops::UnaryOp::exp) {
                    return __float2half_rn(::expf(input));
                } else if constexpr (Op == core::ops::UnaryOp::log) {
                    return __float2half_rn(::logf(input));
                } else if constexpr (Op == core::ops::UnaryOp::sin) {
                    return __float2half_rn(::sinf(input));
                } else {
                    return __float2half_rn(::cosf(input));
                }
            } else if constexpr (std::is_same_v<T, float>) {
                if constexpr (Op == core::ops::UnaryOp::exp) {
                    return ::expf(value);
                } else if constexpr (Op == core::ops::UnaryOp::log) {
                    return ::logf(value);
                } else if constexpr (Op == core::ops::UnaryOp::sin) {
                    return ::sinf(value);
                } else {
                    return ::cosf(value);
                }
            } else {
                if constexpr (Op == core::ops::UnaryOp::exp) {
                    return ::exp(value);
                } else if constexpr (Op == core::ops::UnaryOp::log) {
                    return ::log(value);
                } else if constexpr (Op == core::ops::UnaryOp::sin) {
                    return ::sin(value);
                } else {
                    return ::cos(value);
                }
            }
        }

        template <typename T, core::ops::UnaryOp Op>
        __global__ void unary_contiguous_kernel(T* out, const T* input, std::size_t elements) {
            std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                out[linear] = apply_unary<T, Op>(input[linear]);
            }
        }

        template <typename T, core::ops::UnaryOp Op>
        __global__ void unary_strided_kernel(
            T* out,
            const T* input,
            std::size_t elements,
            elementwise::Index index
        ) {
            std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                const auto input_offset = elementwise::storage_offset(linear, index);
                out[linear] = apply_unary<T, Op>(input[input_offset]);
            }
        }

        template <typename T, core::ops::UnaryOp Op>
        void launch_unary(core::Tensor& out, const core::Tensor& input, cudaStream_t stream) {
            const auto elements = out.numel();
            auto* out_data = static_cast<T*>(out.data());
            const auto* input_data = static_cast<const T*>(input.data());
            const auto grid = elementwise::blocks(elements);

            if (input.is_contiguous()) {
                unary_contiguous_kernel<T, Op>
                    <<<grid, elementwise::threads, 0, stream>>>(out_data, input_data, elements);
            } else {
                const auto index = elementwise::make_index(input);
                unary_strided_kernel<T, Op>
                    <<<grid, elementwise::threads, 0, stream>>>(out_data, input_data, elements, index);
            }

            check(cudaGetLastError(), "CUDA pointwise unary kernel");
        }

        template <typename T>
        void launch_floating_unary(
            core::ops::UnaryOp op,
            core::Tensor& out,
            const core::Tensor& input,
            cudaStream_t stream
        ) {
            switch (op) {
                case core::ops::UnaryOp::rsqrt:
                    launch_unary<T, core::ops::UnaryOp::rsqrt>(out, input, stream);
                    return;
                case core::ops::UnaryOp::abs:
                    launch_unary<T, core::ops::UnaryOp::abs>(out, input, stream);
                    return;
                case core::ops::UnaryOp::sign:
                    launch_unary<T, core::ops::UnaryOp::sign>(out, input, stream);
                    return;
                case core::ops::UnaryOp::positive:
                    launch_unary<T, core::ops::UnaryOp::positive>(out, input, stream);
                    return;
                case core::ops::UnaryOp::relu:
                    launch_unary<T, core::ops::UnaryOp::relu>(out, input, stream);
                    return;
                case core::ops::UnaryOp::tanh:
                    launch_unary<T, core::ops::UnaryOp::tanh>(out, input, stream);
                    return;
                case core::ops::UnaryOp::sigmoid:
                    launch_unary<T, core::ops::UnaryOp::sigmoid>(out, input, stream);
                    return;
                case core::ops::UnaryOp::softplus:
                    launch_unary<T, core::ops::UnaryOp::softplus>(out, input, stream);
                    return;
                case core::ops::UnaryOp::erf:
                    launch_unary<T, core::ops::UnaryOp::erf>(out, input, stream);
                    return;
                case core::ops::UnaryOp::log1p:
                    launch_unary<T, core::ops::UnaryOp::log1p>(out, input, stream);
                    return;
                case core::ops::UnaryOp::expm1:
                    launch_unary<T, core::ops::UnaryOp::expm1>(out, input, stream);
                    return;
                case core::ops::UnaryOp::neg:
                    launch_unary<T, core::ops::UnaryOp::neg>(out, input, stream);
                    return;
                case core::ops::UnaryOp::exp:
                    launch_unary<T, core::ops::UnaryOp::exp>(out, input, stream);
                    return;
                case core::ops::UnaryOp::log:
                    launch_unary<T, core::ops::UnaryOp::log>(out, input, stream);
                    return;
                case core::ops::UnaryOp::sin:
                    launch_unary<T, core::ops::UnaryOp::sin>(out, input, stream);
                    return;
                case core::ops::UnaryOp::cos:
                    launch_unary<T, core::ops::UnaryOp::cos>(out, input, stream);
                    return;
            }

            ORMI_THROW(OperationError, "Unknown CUDA unary pointwise operation.");
        }
    }

    void pointwise_unary(core::ops::UnaryOp op, core::Tensor& out, const core::Tensor& input) {
        if (out.numel() == 0) {
            return;
        }

        auto& cuda_state = state(out.device());
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

        switch (out.dtype()) {
            case core::DType::float64:
                launch_floating_unary<double>(op, out, input, cuda_state.stream());
                return;
            case core::DType::float32:
                launch_floating_unary<float>(op, out, input, cuda_state.stream());
                return;
            case core::DType::float16:
                launch_floating_unary<__half>(op, out, input, cuda_state.stream());
                return;
            case core::DType::int32:
                if (op != core::ops::UnaryOp::neg) {
                    ORMI_THROW(DTypeError, "This CUDA operation requires floating-point tensors.");
                }
                launch_unary<std::int32_t, core::ops::UnaryOp::neg>(out, input, cuda_state.stream());
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CUDA unary pointwise operations do not support Bool.");
        }

        ORMI_THROW(DTypeError, "CUDA unary pointwise operations cannot handle an unknown dtype.");
    }
}
