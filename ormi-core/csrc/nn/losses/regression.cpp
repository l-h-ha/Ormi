#include <cmath>
#include <memory>
#include <string_view>

#include <ormi/core/ops/pointwise.h>
#include <ormi/nn/losses/regression.h>

#include "nn/functional.h"

namespace ormi::nn {
    using namespace core::ops;

    std::shared_ptr<core::Tensor> l1_loss(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction
    ) {
        const auto [values, targets] = detail::loss_inputs(prediction, target);
        const auto error = sub(values, targets);
        const auto loss = abs(error);

        return detail::reduce(loss, reduction);
    }

    std::shared_ptr<core::Tensor> huber_loss(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        double delta,
        std::string_view reduction
    ) {
        if (!std::isfinite(delta) || delta <= 0) {
            ORMI_THROW(OperationError, "Huber delta must be finite and positive.");
        }

        const auto [values, targets] = detail::loss_inputs(prediction, target);
        const auto error = sub(values, targets);
        const auto threshold = detail::constant(error, delta);
        const auto lower_bound = neg(threshold);
        const auto lower_clipped = maximum(error, lower_bound);
        const auto clipped_error = minimum(lower_clipped, threshold);
        const auto tail_error = sub(abs(error), abs(clipped_error));

        const auto squared_error = mul(clipped_error, clipped_error);
        const auto quadratic_loss = mul(detail::constant(error, 0.5), squared_error);
        const auto linear_loss = mul(threshold, tail_error);
        const auto loss = add(quadratic_loss, linear_loss);

        return detail::reduce(loss, reduction);
    }

    std::shared_ptr<core::Tensor> smooth_l1_loss(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        double beta,
        std::string_view reduction
    ) {
        if (!std::isfinite(beta) || beta < 0) {
            ORMI_THROW(OperationError, "Smooth L1 beta must be finite and nonnegative.");
        }
        if (beta == 0) {
            return l1_loss(prediction, target, reduction);
        }

        const auto loss = huber_loss(prediction, target, beta, "none");
        const auto scaled_loss = div(loss, detail::constant(loss, beta));

        return detail::reduce(scaled_loss, reduction);
    }
}
