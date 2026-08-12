#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <ormi/backend/fusion.h>

namespace ormi::backends::cuda::fusion {
    struct Tile {
        std::size_t axis;
        std::size_t width;
        std::size_t tasks;
        std::vector<std::size_t> output;
        std::vector<std::size_t> reduced;
        std::vector<std::vector<std::size_t>> inputs;
    };

    std::optional<Tile> tile(
        const core::runtime::fusion::Expr& expression
    );
}
