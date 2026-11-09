#pragma once
#include <cstdint>
#include <memory>
#include <string_view>

#include <ormi/core/tensor.h>

namespace ormi::nn {
    std::shared_ptr<core::Tensor> binary_cross_entropy(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction = "mean",
        double eps = 1e-7
    );
    std::shared_ptr<core::Tensor> binary_cross_entropy_with_logits(
        const std::shared_ptr<core::Tensor>& logits,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction = "mean",
        const std::shared_ptr<core::Tensor>& pos_weight = nullptr
    );
    std::shared_ptr<core::Tensor> cross_entropy(
        const std::shared_ptr<core::Tensor>& logits,
        const std::shared_ptr<core::Tensor>& target,
        std::int64_t axis = -1,
        std::string_view reduction = "mean",
        std::int32_t ignore_index = -100,
        double label_smoothing = 0.0
    );
    std::shared_ptr<core::Tensor> nll_loss(
        const std::shared_ptr<core::Tensor>& log_probabilities,
        const std::shared_ptr<core::Tensor>& target,
        std::int64_t axis = -1,
        std::string_view reduction = "mean",
        std::int32_t ignore_index = -100
    );
}
