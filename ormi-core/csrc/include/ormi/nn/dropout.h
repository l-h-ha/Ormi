#pragma once
#include <cstdint>
#include <memory>
#include <optional>

#include <ormi/core/tensor.h>

namespace ormi::nn {
    // Drop each element with probability p and scale retained values by 1/(1-p).
    // Each execution draws a fresh mask. Derivatives reuse that mask.
    std::shared_ptr<core::Tensor> dropout(
        const std::shared_ptr<core::Tensor>& input,
        double p = 0.5,
        bool training = true,
        std::optional<std::uint64_t> seed = std::nullopt
    );
}
