#pragma once

#include <ormi/core/tensor.h>

namespace ormi::core::runtime::linalg {
    void matmul(Tensor& out, const Tensor& lhs, const Tensor& rhs);
}
