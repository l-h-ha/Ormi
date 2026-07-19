#pragma once

#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"

namespace ormi::backends::cuda::fusion {
    void execute(core::Tensor& out, const core::runtime::fusion::Expr& expression);
}
