#pragma once

#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"

namespace ormi::backends::cpu::jit {
    void execute(core::Tensor& out, const core::runtime::fusion::Expr& expression);
}
