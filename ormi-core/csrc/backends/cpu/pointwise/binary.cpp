#include <cstdint>
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
        void pointwise_binary_typed(
            core::Tensor& out,
            const core::Tensor& lhs,
            const core::Tensor& rhs
        ) {
            auto* out_data = static_cast<T*>(out.data());
            const auto* lhs_data = static_cast<const T*>(lhs.data());
            const auto* rhs_data = static_cast<const T*>(rhs.data());
            const auto elements = out.numel();

            if (
                out.is_contiguous() &&
                lhs.is_contiguous() &&
                rhs.is_contiguous() &&
                lhs.shape() == out.shape() &&
                rhs.shape() == out.shape()
            ) {
                for (std::size_t i = 0; i < elements; ++i) {
                    out_data[i] = apply_binary<T, Op>(lhs_data[i], rhs_data[i]);
                }
                return;
            }

            const auto& out_dims = out.shape().dims();
            const auto lhs_strides = aligned_broadcast_strides(out.shape(), lhs);
            const auto rhs_strides = aligned_broadcast_strides(out.shape(), rhs);

            for (std::size_t i = 0; i < elements; ++i) {
                std::size_t linear = i;
                std::size_t lhs_offset = 0;
                std::size_t rhs_offset = 0;

                for (std::size_t axis = out.rank(); axis > 0; --axis) {
                    const auto index = axis - 1;
                    const auto coord = linear % static_cast<std::size_t>(out_dims[index]);
                    linear /= static_cast<std::size_t>(out_dims[index]);
                    lhs_offset += coord * lhs_strides[index];
                    rhs_offset += coord * rhs_strides[index];
                }

                out_data[i] = apply_binary<T, Op>(
                    lhs_data[lhs_offset],
                    rhs_data[rhs_offset]
                );
            }
        }

        template <core::ops::BinaryOp Op>
        void pointwise_binary_op(
            core::Tensor& out,
            const core::Tensor& lhs,
            const core::Tensor& rhs
        ) {
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
                    ORMI_THROW(
                        DTypeError,
                        "CPU pointwise binary operations do not support Bool."
                    );
            }

            ORMI_THROW(
                DTypeError,
                "CPU pointwise binary operations cannot handle an unknown dtype."
            );
        }
    }

    void pointwise_binary(
        core::ops::BinaryOp op,
        core::Tensor& out,
        const core::Tensor& lhs,
        const core::Tensor& rhs
    ) {
        switch (op) {
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
