#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/index.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/nn/activation.h>
#include <ormi/nn/losses/classification.h>

#include "nn/functional.h"

namespace ormi::nn {
    using namespace core::ops;

    namespace {
        std::shared_ptr<core::Tensor> valid_targets(
            const std::shared_ptr<core::Tensor>& target,
            std::int32_t ignore_index,
            core::DType dtype
        ) {
            // Float64 represents every Int32 label exactly, including large ignore values.
            const auto labels = cast(target, core::DType::float64);
            const auto ignored_label = detail::constant(labels, ignore_index);
            const auto distance = abs(sub(labels, ignored_label));
            const auto mask = positive(distance);

            return cast(mask, dtype);
        }

        std::shared_ptr<core::Tensor> reduce_indexed(
            const std::shared_ptr<core::Tensor>& loss,
            const std::shared_ptr<core::Tensor>& valid,
            std::string_view reduction
        ) {
            detail::check_reduction(reduction);
            if (reduction == "none") {
                return loss;
            }

            const auto total = sum(loss);
            if (reduction == "sum") {
                return total;
            }

            const auto valid_count = sum(valid);
            return div(total, valid_count);
        }
    }

    std::shared_ptr<core::Tensor> binary_cross_entropy(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction,
        double eps
    ) {
        if (!std::isfinite(eps) || eps <= 0 || eps >= 0.5) {
            ORMI_THROW(OperationError, "BCE eps must be finite and between 0 and 0.5.");
        }

        const auto [values, targets] = detail::loss_inputs(prediction, target);
        double smallest_probability = std::numeric_limits<float>::min();
        double largest_probability = std::nextafter(1.0f, 0.0f);
        if (values->dtype() == core::DType::float64) {
            smallest_probability = std::numeric_limits<double>::min();
            largest_probability = std::nextafter(1.0, 0.0);
        }

        const auto lower = detail::constant(values, std::max(eps, smallest_probability));
        const auto upper = detail::constant(values, std::min(1 - eps, largest_probability));
        const auto lower_clipped = maximum(values, lower);
        const auto probability = minimum(lower_clipped, upper);

        const auto positive_term = mul(targets, log(probability));
        const auto negative_targets = sub(detail::constant(values, 1), targets);
        const auto log_complement = log1p(neg(probability));
        const auto negative_term = mul(negative_targets, log_complement);
        const auto loss = neg(add(positive_term, negative_term));

        return detail::reduce(loss, reduction);
    }

    std::shared_ptr<core::Tensor> binary_cross_entropy_with_logits(
        const std::shared_ptr<core::Tensor>& logits,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction,
        const std::shared_ptr<core::Tensor>& pos_weight
    ) {
        const auto [values, targets] = detail::loss_inputs(logits, target);
        const auto positive_penalty = core::ops::softplus(neg(values));
        auto positive_loss = mul(targets, positive_penalty);
        if (pos_weight) {
            const auto weight = detail::floating(pos_weight);
            if (weight->device() != values->device()) {
                ORMI_THROW(DeviceError, "BCE pos_weight must use the logits device.");
            }
            if (core::broadcast(weight->shape(), values->shape()) != values->shape()) {
                ORMI_THROW(ShapeError, "BCE pos_weight must broadcast to the logits shape.");
            }
            positive_loss = mul(positive_loss, cast(weight, values->dtype()));
        }

        const auto negative_targets = sub(detail::constant(values, 1), targets);
        const auto negative_penalty = core::ops::softplus(values);
        const auto negative_loss = mul(negative_targets, negative_penalty);
        const auto loss = add(positive_loss, negative_loss);

        return detail::reduce(loss, reduction);
    }

    std::shared_ptr<core::Tensor> nll_loss(
        const std::shared_ptr<core::Tensor>& log_probabilities,
        const std::shared_ptr<core::Tensor>& target,
        std::int64_t axis,
        std::string_view reduction,
        std::int32_t ignore_index
    ) {
        const auto value = detail::floating(log_probabilities);
        if (!target) {
            ORMI_THROW(OperationError, "NLL requires target indices.");
        }
        axis = detail::axis(*value, axis);

        const auto selected = pick(value, target, axis, ignore_index);
        const auto loss = neg(selected);
        const auto valid = valid_targets(target, ignore_index, value->dtype());

        return reduce_indexed(loss, valid, reduction);
    }

    std::shared_ptr<core::Tensor> cross_entropy(
        const std::shared_ptr<core::Tensor>& logits,
        const std::shared_ptr<core::Tensor>& target,
        std::int64_t axis,
        std::string_view reduction,
        std::int32_t ignore_index,
        double label_smoothing
    ) {
        if (!std::isfinite(label_smoothing) || label_smoothing < 0 || label_smoothing > 1) {
            ORMI_THROW(OperationError, "Label smoothing must be between 0 and 1.");
        }
        const auto value = detail::floating(logits);
        if (!target) {
            ORMI_THROW(OperationError, "Cross entropy requires a target tensor.");
        }
        axis = detail::axis(*value, axis);

        const auto log_probabilities = nn::log_softmax(value, axis);
        if (core::is_floating(target->dtype())) {
            const auto [log_values, targets] = detail::loss_inputs(log_probabilities, target);
            auto distribution = targets;

            if (label_smoothing != 0) {
                const auto class_count = static_cast<double>(value->shape()[static_cast<std::size_t>(axis)]);
                const auto target_scale = detail::constant(targets, 1 - label_smoothing);
                const auto scaled_targets = mul(targets, target_scale);
                const auto uniform_weight = detail::constant(targets, label_smoothing / class_count);
                distribution = add(scaled_targets, uniform_weight);
            }

            const auto weighted_logs = mul(distribution, log_values);
            const auto loss = neg(sum(weighted_logs, {axis}));

            return detail::reduce(loss, reduction);
        }

        const auto valid = valid_targets(target, ignore_index, value->dtype());
        const auto selected = pick(log_probabilities, target, axis, ignore_index);
        auto loss = neg(selected);

        if (label_smoothing != 0) {
            const auto uniform_loss = neg(mean(log_probabilities, {axis}));
            const auto valid_uniform_loss = mul(uniform_loss, valid);
            const auto target_loss = mul(loss, detail::constant(loss, 1 - label_smoothing));
            const auto smoothing_loss = mul(valid_uniform_loss, detail::constant(loss, label_smoothing));
            loss = add(target_loss, smoothing_loss);
        }

        return reduce_indexed(loss, valid, reduction);
    }
}
