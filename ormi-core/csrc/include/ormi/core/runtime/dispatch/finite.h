#pragma once
#include <span>

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::runtime::finite {
    bool all(std::span<const Tensor* const> inputs);
}
