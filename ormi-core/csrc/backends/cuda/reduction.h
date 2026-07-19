#pragma once
#include <cstdint>
#include <span>

#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda {
    void reduce(
        core::ops::ReduceOp op,
        core::Tensor& out,
        const core::Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    );
}
