#pragma once

#include <memory>
#include <span>

#include <ormi/backend/fusion.h>
#include <ormi/core/op.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/compile.h"

namespace ormi::backends::cuda {
    class State;
}

namespace ormi::backends::cuda::fusion::execution {
    core::Setup prepare(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        State& cuda_state,
        std::shared_ptr<Compiled> compiled
    );
    void run(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        State& cuda_state,
        std::shared_ptr<Compiled> compiled
    );
}
