#pragma once
#include <span>

namespace ormi::core {
    class Tensor;
}

namespace ormi::backends::cuda {
    bool all_finite(std::span<const core::Tensor* const> inputs);
}
