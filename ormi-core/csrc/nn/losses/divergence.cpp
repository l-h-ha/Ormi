#include <cmath>
#include <limits>
#include <memory>
#include <string_view>

#include <ormi/core/ops/pointwise.h>
#include <ormi/nn/losses/divergence.h>

#include "nn/functional.h"

namespace ormi::nn {
    std::shared_ptr<core::Tensor> kl_div(
        const std::shared_ptr<core::Tensor>& log_probabilities,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction,
        bool log_target
    ) {
        using namespace core::ops;

        const auto [log_values, targets] = detail::loss_inputs(log_probabilities, target);
        double smallest_probability = std::numeric_limits<float>::min();
        if (log_values->dtype() == core::DType::float64) {
            smallest_probability = std::numeric_limits<double>::min();
        }

        std::shared_ptr<core::Tensor> loss;
        if (log_target) {
            const auto lower_bound = detail::constant(targets, std::log(smallest_probability));
            const auto log_targets = maximum(targets, lower_bound);
            const auto log_ratio = sub(log_targets, log_values);
            loss = mul(exp(targets), log_ratio);
        } else {
            const auto lower_bound = detail::constant(targets, smallest_probability);
            const auto clamped_targets = maximum(targets, lower_bound);
            const auto log_targets = log(clamped_targets);
            const auto log_ratio = sub(log_targets, log_values);
            loss = mul(targets, log_ratio);
        }

        if (reduction != "batchmean") {
            return detail::reduce(loss, reduction);
        }
        if (log_values->rank() == 0) {
            ORMI_THROW(ShapeError, "KL batchmean requires a batch axis.");
        }

        const auto batch_size = detail::constant(loss, static_cast<double>(log_values->shape()[0]));
        const auto total = sum(loss);

        return div(total, batch_size);
    }
}
