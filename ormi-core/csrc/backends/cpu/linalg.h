#pragma once

#include <ormi/core/tensor.h>

namespace ormi::backends::cpu {
    void matmul(core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs);
}
