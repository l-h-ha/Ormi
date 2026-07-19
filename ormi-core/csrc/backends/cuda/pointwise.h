#pragma once

#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda {
    void pointwise_binary(
        core::ops::BinaryOp op,
        core::Tensor& out,
        const core::Tensor& lhs,
        const core::Tensor& rhs
    );

    void pointwise_unary(
        core::ops::UnaryOp op,
        core::Tensor& out,
        const core::Tensor& input
    );
}
