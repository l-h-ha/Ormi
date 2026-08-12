#pragma once
#include <span>
#include <string>

#include <ormi/backend/fusion.h>

#include "backends/cpu/jit/loop.h"

namespace ormi::backends::cpu::jit {
    std::string emit(const core::runtime::fusion::Expr& expression);
    std::string emit(
        const core::runtime::fusion::Expr& expression,
        const Loop& loop
    );
    std::string emit(
        std::span<const core::runtime::fusion::Expr* const> expressions,
        std::span<const Loop> loops
    );
}
