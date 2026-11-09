#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/pointwise.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename T, core::ops::BinaryOp Op>
        T apply_binary(T lhs, T rhs) {
            if constexpr (Op == core::ops::BinaryOp::add) {
                return lhs + rhs;
            } else if constexpr (Op == core::ops::BinaryOp::sub) {
                return lhs - rhs;
            } else if constexpr (Op == core::ops::BinaryOp::mul) {
                return lhs * rhs;
            } else if constexpr (Op == core::ops::BinaryOp::minimum || Op == core::ops::BinaryOp::maximum) {
                using Value = std::conditional_t<std::is_same_v<T, core::Float16>, float, T>;
                const auto left = static_cast<Value>(lhs), right = static_cast<Value>(rhs);
                if (left != left)
                    return lhs;
                if (right != right)
                    return rhs;
                if constexpr (Op == core::ops::BinaryOp::minimum)
                    return left < right ? lhs : rhs;
                else
                    return left > right ? lhs : rhs;
            } else {
                return lhs / rhs;
            }
        }

        std::vector<std::size_t> aligned_broadcast_strides(
            const core::Shape& out_shape,
            const core::Tensor& tensor
        ) {
            const auto& in_dims = tensor.shape().dims();
            const auto strides = tensor.strides();
            const auto out_rank = out_shape.rank();
            const auto in_rank = in_dims.size();
            const auto rank_offset = out_rank - in_rank;
            std::vector<std::size_t> aligned(out_rank, 0);

            for (std::size_t axis = rank_offset; axis < out_rank; ++axis) {
                const auto in_axis = axis - rank_offset;
                if (in_dims[in_axis] != 1) {
                    aligned[axis] = static_cast<std::size_t>(strides[in_axis]);
                }
            }

            return aligned;
        }

        template <typename T, core::ops::BinaryOp Op>
        void binary_loop(
            T* out,
            const T* lhs,
            const T* rhs,
            std::size_t elements,
            std::size_t lhs_stride,
            std::size_t rhs_stride
        ) {
            if (lhs_stride == 1 && rhs_stride == 1) {
                for (std::size_t i = 0; i < elements; ++i) {
                    out[i] = apply_binary<T, Op>(lhs[i], rhs[i]);
                }
            } else if (lhs_stride == 0 && rhs_stride == 1) {
                const auto value = lhs[0];
                for (std::size_t i = 0; i < elements; ++i) {
                    out[i] = apply_binary<T, Op>(value, rhs[i]);
                }
            } else if (lhs_stride == 1 && rhs_stride == 0) {
                const auto value = rhs[0];
                for (std::size_t i = 0; i < elements; ++i) {
                    out[i] = apply_binary<T, Op>(lhs[i], value);
                }
            } else if (lhs_stride == 0 && rhs_stride == 0) {
                std::fill_n(out, elements, apply_binary<T, Op>(lhs[0], rhs[0]));
            } else {
                for (std::size_t i = 0; i < elements; ++i) {
                    out[i] = apply_binary<T, Op>(lhs[i * lhs_stride], rhs[i * rhs_stride]);
                }
            }
        }

        template <typename T, core::ops::BinaryOp Op>
        void pointwise_binary_typed(core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
            auto* out_data = static_cast<T*>(out.data());
            const auto* lhs_data = static_cast<const T*>(lhs.data());
            const auto* rhs_data = static_cast<const T*>(rhs.data());
            const auto elements = out.numel();

            if (out.is_contiguous() && lhs.is_contiguous() && rhs.is_contiguous() &&
                lhs.shape() == out.shape() && rhs.shape() == out.shape()) {
                for (std::size_t i = 0; i < elements; ++i) {
                    out_data[i] = apply_binary<T, Op>(lhs_data[i], rhs_data[i]);
                }
                return;
            }

            if (elements == 0) {
                return;
            }
            if (lhs.numel() == 1 && rhs.is_contiguous() && rhs.shape() == out.shape()) {
                binary_loop<T, Op>(out_data, lhs_data, rhs_data, elements, 0, 1);
                return;
            }
            if (rhs.numel() == 1 && lhs.is_contiguous() && lhs.shape() == out.shape()) {
                binary_loop<T, Op>(out_data, lhs_data, rhs_data, elements, 1, 0);
                return;
            }

            const auto& out_dims = out.shape().dims();
            const auto lhs_strides = aligned_broadcast_strides(out.shape(), lhs);
            const auto rhs_strides = aligned_broadcast_strides(out.shape(), rhs);

            // Combine trailing axes while both inputs keep a constant step.
            std::size_t span = 1;
            std::size_t lhs_step = 0;
            std::size_t rhs_step = 0;
            auto outer_axes = out.rank();
            while (outer_axes > 0) {
                const auto axis = outer_axes - 1;
                const auto extent = static_cast<std::size_t>(out_dims[axis]);
                if (extent == 1) {
                    --outer_axes;
                    continue;
                }
                if (span == 1) {
                    lhs_step = lhs_strides[axis];
                    rhs_step = rhs_strides[axis];
                } else if (lhs_strides[axis] != lhs_step * span || rhs_strides[axis] != rhs_step * span) {
                    break;
                }
                span *= extent;
                --outer_axes;
            }

            const auto rows = elements / span;
            for (std::size_t row = 0; row < rows; ++row) {
                std::size_t linear = row;
                std::size_t lhs_offset = 0;
                std::size_t rhs_offset = 0;

                for (std::size_t axis = outer_axes; axis > 0; --axis) {
                    const auto output_axis = axis - 1;
                    const auto extent = static_cast<std::size_t>(out_dims[output_axis]);
                    const auto coordinate = linear % extent;
                    linear /= extent;
                    lhs_offset += coordinate * lhs_strides[output_axis];
                    rhs_offset += coordinate * rhs_strides[output_axis];
                }

                binary_loop<T, Op>(
                    out_data + row * span,
                    lhs_data + lhs_offset,
                    rhs_data + rhs_offset,
                    span,
                    lhs_step,
                    rhs_step
                );
            }
        }

        template <core::ops::BinaryOp Op>
        void pointwise_binary_op(core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
            switch (out.dtype()) {
                case core::DType::float64:
                    pointwise_binary_typed<double, Op>(out, lhs, rhs);
                    return;
                case core::DType::float32:
                    pointwise_binary_typed<float, Op>(out, lhs, rhs);
                    return;
                case core::DType::float16:
                    pointwise_binary_typed<core::Float16, Op>(out, lhs, rhs);
                    return;
                case core::DType::int32:
                    pointwise_binary_typed<std::int32_t, Op>(out, lhs, rhs);
                    return;
                case core::DType::boolean:
                    ORMI_THROW(DTypeError, "CPU pointwise binary operations do not support Bool.");
            }

            ORMI_THROW(DTypeError, "CPU pointwise binary operations cannot handle an unknown dtype.");
        }
    }

    void pointwise_binary(
        core::ops::BinaryOp op,
        core::Tensor& out,
        const core::Tensor& lhs,
        const core::Tensor& rhs
    ) {
        switch (op) {
            case core::ops::BinaryOp::minimum:
                pointwise_binary_op<core::ops::BinaryOp::minimum>(out, lhs, rhs);
                return;
            case core::ops::BinaryOp::maximum:
                pointwise_binary_op<core::ops::BinaryOp::maximum>(out, lhs, rhs);
                return;
            case core::ops::BinaryOp::add:
                pointwise_binary_op<core::ops::BinaryOp::add>(out, lhs, rhs);
                return;
            case core::ops::BinaryOp::sub:
                pointwise_binary_op<core::ops::BinaryOp::sub>(out, lhs, rhs);
                return;
            case core::ops::BinaryOp::mul:
                pointwise_binary_op<core::ops::BinaryOp::mul>(out, lhs, rhs);
                return;
            case core::ops::BinaryOp::div:
                pointwise_binary_op<core::ops::BinaryOp::div>(out, lhs, rhs);
                return;
        }

        ORMI_THROW(OperationError, "Unknown binary pointwise op.");
    }
}
