#pragma once
#include <memory>
#include <string_view>

#include <ormi/core/tensor.h>

namespace ormi::nn {
    std::shared_ptr<core::Tensor> kl_div(
        const std::shared_ptr<core::Tensor>& log_probabilities,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction = "batchmean",
        bool log_target = false
    );
}
