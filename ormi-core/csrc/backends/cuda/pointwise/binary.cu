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
        template <typename T, core::ops::BinaryOp Op>
        __device__ T apply_binary(T lhs, T rhs) {
            if constexpr (std::is_same_v<T, __half>) {
                if constexpr (Op == core::ops::BinaryOp::add) {
                    return __hadd(lhs, rhs);
                } else if constexpr (Op == core::ops::BinaryOp::sub) {
                    return __hsub(lhs, rhs);
                } else if constexpr (Op == core::ops::BinaryOp::mul) {
                    return __hmul(lhs, rhs);
                } else if constexpr (Op == core::ops::BinaryOp::minimum ||
                                     Op == core::ops::BinaryOp::maximum) {
                    const float left = __half2float(lhs), right = __half2float(rhs);
                    const auto value = left != left
                                           ? left
                                           : (right != right ? right
                                                             : (Op == core::ops::BinaryOp::minimum
                                                                    ? (left < right ? left : right)
                                                                    : (left > right ? left : right)));
                    return __float2half_rn(value);
                } else {
                    return __hdiv(lhs, rhs);
                }
            } else {
                if constexpr (Op == core::ops::BinaryOp::add) {
                    return lhs + rhs;
                } else if constexpr (Op == core::ops::BinaryOp::sub) {
                    return lhs - rhs;
                } else if constexpr (Op == core::ops::BinaryOp::mul) {
                    return lhs * rhs;
                } else if constexpr (Op == core::ops::BinaryOp::minimum) {
                    return lhs != lhs ? lhs : (rhs != rhs ? rhs : (lhs < rhs ? lhs : rhs));
                } else if constexpr (Op == core::ops::BinaryOp::maximum) {
                    return lhs != lhs ? lhs : (rhs != rhs ? rhs : (lhs > rhs ? lhs : rhs));
                } else {
                    return lhs / rhs;
                }
            }
        }

        template <typename T, core::ops::BinaryOp Op>
        __global__ void binary_contiguous_kernel(T* out, const T* lhs, const T* rhs, std::size_t elements) {
            std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                out[linear] = apply_binary<T, Op>(lhs[linear], rhs[linear]);
            }
        }

        template <typename T, core::ops::BinaryOp Op>
        __global__ void binary_strided_kernel(
            T* out,
            const T* lhs,
            const T* rhs,
            std::size_t elements,
            elementwise::BinaryIndex index
        ) {
            std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                std::size_t lhs_offset;
                std::size_t rhs_offset;
                elementwise::storage_offsets(linear, index, lhs_offset, rhs_offset);
                out[linear] = apply_binary<T, Op>(lhs[lhs_offset], rhs[rhs_offset]);
            }
        }

        template <typename T, core::ops::BinaryOp Op>
        void launch_binary(
            core::Tensor& out,
            const core::Tensor& lhs,
            const core::Tensor& rhs,
            cudaStream_t stream
        ) {
            const auto elements = out.numel();
            auto* out_data = static_cast<T*>(out.data());
            const auto* lhs_data = static_cast<const T*>(lhs.data());
            const auto* rhs_data = static_cast<const T*>(rhs.data());
            const auto grid = elementwise::blocks(elements);

            if (lhs.is_contiguous() && rhs.is_contiguous() && lhs.shape() == out.shape() &&
                rhs.shape() == out.shape()) {
                binary_contiguous_kernel<T, Op>
                    <<<grid, elementwise::threads, 0, stream>>>(out_data, lhs_data, rhs_data, elements);
            } else {
                const auto index = elementwise::make_index(out, lhs, rhs);
                binary_strided_kernel<T, Op><<<grid, elementwise::threads, 0, stream>>>(
                    out_data,
                    lhs_data,
                    rhs_data,
                    elements,
                    index
                );
            }

            check(cudaGetLastError(), "CUDA pointwise binary kernel");
        }

        template <typename T>
        void launch_binary(
            core::ops::BinaryOp op,
            core::Tensor& out,
            const core::Tensor& lhs,
            const core::Tensor& rhs,
            cudaStream_t stream
        ) {
            switch (op) {
                case core::ops::BinaryOp::minimum:
                    launch_binary<T, core::ops::BinaryOp::minimum>(out, lhs, rhs, stream);
                    return;
                case core::ops::BinaryOp::maximum:
                    launch_binary<T, core::ops::BinaryOp::maximum>(out, lhs, rhs, stream);
                    return;
                case core::ops::BinaryOp::add:
                    launch_binary<T, core::ops::BinaryOp::add>(out, lhs, rhs, stream);
                    return;
                case core::ops::BinaryOp::sub:
                    launch_binary<T, core::ops::BinaryOp::sub>(out, lhs, rhs, stream);
                    return;
                case core::ops::BinaryOp::mul:
                    launch_binary<T, core::ops::BinaryOp::mul>(out, lhs, rhs, stream);
                    return;
                case core::ops::BinaryOp::div:
                    launch_binary<T, core::ops::BinaryOp::div>(out, lhs, rhs, stream);
                    return;
            }

            ORMI_THROW(OperationError, "Unknown CUDA binary pointwise operation.");
        }
    }

    void pointwise_binary(
        core::ops::BinaryOp op,
        core::Tensor& out,
        const core::Tensor& lhs,
        const core::Tensor& rhs
    ) {
        if (out.numel() == 0) {
            return;
        }

        auto& cuda_state = state(out.device());
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

        switch (out.dtype()) {
            case core::DType::float64:
                launch_binary<double>(op, out, lhs, rhs, cuda_state.stream());
                return;
            case core::DType::float32:
                launch_binary<float>(op, out, lhs, rhs, cuda_state.stream());
                return;
            case core::DType::float16:
                launch_binary<__half>(op, out, lhs, rhs, cuda_state.stream());
                return;
            case core::DType::int32:
                launch_binary<std::int32_t>(op, out, lhs, rhs, cuda_state.stream());
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CUDA pointwise binary operations do not support Bool.");
        }

        ORMI_THROW(DTypeError, "CUDA pointwise binary operations cannot handle an unknown dtype.");
    }
}
