#pragma once
#include <cstdint>
#include <span>

#include <ormi/core/tensor.h>

namespace ormi::backends::cuda {
    void reduce_max(core::Tensor& out, const core::Tensor& input, std::span<const std::int64_t> axes);
}
