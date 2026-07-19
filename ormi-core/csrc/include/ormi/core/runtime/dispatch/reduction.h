#pragma once

#include <cstdint>
#include <span>

#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

namespace ormi::core::runtime::reduction {
    void reduce(
        ops::ReduceOp op,
        Tensor& out,
        const Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    );
}
