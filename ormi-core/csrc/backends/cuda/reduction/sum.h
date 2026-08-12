#pragma once
#include <cstdint>
#include <span>

namespace ormi::core {
    class Tensor;
}

namespace ormi::backends::cuda {
    void reduce_sum(
        core::Tensor& out,
        const core::Tensor& input,
        std::span<const std::int64_t> axes
    );

    void reduce_mean(
        core::Tensor& out,
        const core::Tensor& input,
        std::span<const std::int64_t> axes
    );
}
