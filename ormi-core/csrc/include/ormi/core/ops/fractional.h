#pragma once
#include <cstdint>
#include <memory>

#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    // Left-sided Caputo derivative on a uniform grid whose first sample is
    // the history origin. L1 covers 0 < alpha < 1. L2 covers 1 < alpha < 2
    // and requires the first derivative at the origin, shaped like input
    // with the history axis removed.
    std::shared_ptr<Tensor> caputo(
        const std::shared_ptr<Tensor>& input,
        double alpha,
        double spacing,
        std::int64_t axis = -1,
        const std::shared_ptr<Tensor>& initial_derivative = nullptr
    );

    bool is_caputo(const Tensor& tensor);
}
