#pragma once

#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

namespace ormi::core::runtime::pointwise {
    void binary(ops::BinaryOp op, Tensor& out, const Tensor& lhs, const Tensor& rhs);
    void unary(ops::UnaryOp op, Tensor& out, const Tensor& t);
}
