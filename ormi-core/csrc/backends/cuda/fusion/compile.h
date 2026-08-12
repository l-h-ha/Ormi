#pragma once

#include <memory>
#include <optional>

#include <cuda.h>

#include <ormi/backend/fusion.h>

#include "backends/cuda/fusion/tile.h"

namespace ormi::backends::cuda {
    class State;
}

namespace ormi::backends::cuda::fusion {
    struct Compiled {
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        CUfunction finish = nullptr;
        int threads = 0;
        int max_blocks = 0;
        std::optional<Tile> tiled;
    };

    std::shared_ptr<Compiled> compile(
        const core::runtime::fusion::Expr& expression,
        const State& cuda_state
    );
}
