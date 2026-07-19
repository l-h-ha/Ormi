#pragma once
#include <string>

#include "core/runtime/fusion/expr.h"

namespace ormi::backends::cuda::fusion {
    std::string emit(const core::runtime::fusion::Expr& expression);
}
