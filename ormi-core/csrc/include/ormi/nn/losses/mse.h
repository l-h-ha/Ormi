#pragma once
#include <memory>
#include <string_view>

#include <ormi/core/tensor.h>

namespace ormi::nn {
    std::shared_ptr<core::Tensor> mse_loss(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction = "mean"
    );
}
