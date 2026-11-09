#pragma once
#include <memory>
#include <string_view>

#include <ormi/core/tensor.h>

namespace ormi::nn {
    std::shared_ptr<core::Tensor> l1_loss(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction = "mean"
    );
    std::shared_ptr<core::Tensor> huber_loss(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        double delta = 1.0,
        std::string_view reduction = "mean"
    );
    std::shared_ptr<core::Tensor> smooth_l1_loss(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        double beta = 1.0,
        std::string_view reduction = "mean"
    );
}
