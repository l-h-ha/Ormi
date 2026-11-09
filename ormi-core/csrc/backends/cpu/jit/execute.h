#pragma once

#include <memory>
#include <span>

#include <ormi/backend/fusion.h>
#include <ormi/core/setup.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/compile.h"

namespace ormi::backends::cpu::jit::execution {
    core::Setup prepare(
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        std::shared_ptr<Compiled> compiled
    );
    void run(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        const Compiled& compiled
    );
}
