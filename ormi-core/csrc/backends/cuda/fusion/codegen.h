#pragma once
#include <string>

#include <ormi/backend/fusion.h>

namespace ormi::backends::cuda::fusion {
    std::string emit(const core::runtime::fusion::Expr& expression);
}
