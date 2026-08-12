#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <ormi/backend/fusion.h>

namespace ormi::backends::cpu::jit {
    enum class Access {
        indexed,
        linear,
        scalar,
        span,
        fixed
    };

    struct Loop {
        bool vectorized = false;
        bool math = false;
        bool strided = false;
        bool ranged = false;
        bool partial = false;
        std::size_t unit = 0;
        std::size_t span = 0;
        std::size_t lanes = 0;
        std::size_t tile = 0;
        std::optional<std::size_t> split;
        std::optional<std::size_t> tile_axis;
        std::vector<Access> inputs;
        std::vector<std::vector<std::size_t>> strides;
        std::vector<std::size_t> output_dimensions;
        std::vector<std::size_t> reduced_dimensions;
    };

    std::size_t parallel_work(
        const core::runtime::fusion::Expr& expression,
        bool paired_math
    ) noexcept;
    Loop plan(const core::runtime::fusion::Expr& expression);
}
