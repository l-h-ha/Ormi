#pragma once

#include <memory>
#include <span>

#include <ormi/backend/fusion.h>
#include <ormi/core/setup.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/module.h"

namespace ormi::backends::cuda {
    class State;
}

namespace ormi::backends::cuda::fusion::execution {
    // Own the compiled module, launch-argument storage, and any reduction workspace in Setup.
    // Tensor pointers are borrowed from the graph owner. Execution reads their current data pointers.
    core::Setup prepare(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        State& cuda_state,
        std::shared_ptr<Compiled> compiled
    );
    // Direct execution without a saved Setup.
    void run(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        State& cuda_state,
        std::shared_ptr<Compiled> compiled
    );
}
