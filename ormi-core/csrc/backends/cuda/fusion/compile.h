#pragma once

#include <memory>

#include <ormi/backend/fusion.h>

#include "backends/cuda/fusion/module.h"

namespace ormi::backends::cuda {
    class State;
}

namespace ormi::backends::cuda::fusion {
    std::shared_ptr<Compiled> compile(
        const core::runtime::fusion::Expr& expression,
        const State& cuda_state
    );
}
