#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/optim/grad_scaler.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/finite.h"
#include "core/optim/sgd_detail.h"

namespace ormi::core::optim {
    namespace {
        Scalar scale_value(double scale, DType dtype) {
            switch (dtype) {
                case DType::float64:
                    return scale;
                case DType::float32: {
                    const auto value = static_cast<float>(scale);
                    if (!std::isfinite(value) || value == 0.0f) {
                        ORMI_THROW(DTypeError, "Gradient scale is not representable as Float32.");
                    }
                    return value;
                }
                case DType::float16: {
                    const auto value = Float16(static_cast<float>(scale));
                    const auto converted = static_cast<float>(value);
                    if (!std::isfinite(converted) || converted == 0.0f) {
                        ORMI_THROW(DTypeError, "Gradient scale is not representable as Float16.");
                    }
                    return value;
                }
                case DType::int32:
                case DType::boolean:
                    ORMI_THROW(DTypeError, "Loss scaling supports floating-point tensors only.");
            }

            ORMI_THROW(DTypeError, "Loss scaling cannot handle an unknown dtype.");
        }
    }

    GradScaler::GradScaler(double scale) : GradScaler(scale, ScalePolicy{}) {}

    GradScaler::GradScaler(double scale, ScalePolicy policy)
        : scale_(scale), policy_(policy), finite_steps_(0) {
        if (!std::isfinite(scale_) || scale_ < 1.0) {
            ORMI_THROW(OperationError, "Gradient scale must be finite and at least one.");
        }

        if (!std::isfinite(policy_.growth) || policy_.growth <= 1.0) {
            ORMI_THROW(
                OperationError,
                "Gradient scale growth must be finite and greater than one."
            );
        }

        if (
            !std::isfinite(policy_.backoff) ||
            policy_.backoff <= 0.0 ||
            policy_.backoff >= 1.0
        ) {
            ORMI_THROW(
                OperationError,
                "Gradient scale backoff must be finite and between zero and one."
            );
        }

        if (policy_.interval == 0) {
            ORMI_THROW(OperationError, "Gradient scale growth interval must be positive.");
        }

        if (!std::isfinite(policy_.limit) || policy_.limit < scale_) {
            ORMI_THROW(
                OperationError,
                "Gradient scale limit must be finite and at least the initial scale."
            );
        }
    }

    double GradScaler::value() const noexcept {
        return scale_;
    }

    std::shared_ptr<Tensor> GradScaler::scale(const std::shared_ptr<Tensor>& loss) const {
        if (!loss) {
            ORMI_THROW(OperationError, "Loss scaling requires a loss tensor.");
        }

        if (!is_floating(loss->dtype())) {
            ORMI_THROW(DTypeError, "Loss scaling supports floating-point tensors only.");
        }

        if (scale_ == 1.0) {
            return loss;
        }

        const auto factor = ops::full({}, scale_value(scale_, loss->dtype()), loss->device());
        return ops::mul(loss, factor);
    }

    std::optional<std::shared_ptr<Tensor>> GradScaler::step(
        const std::shared_ptr<Tensor>& parameter,
        const std::shared_ptr<Tensor>& scaled_gradient,
        double learning_rate
    ) {
        const auto step_size = detail::check_sgd(
            parameter,
            scaled_gradient,
            learning_rate,
            scale_
        );
        const std::array<const Tensor*, 1> gradients{scaled_gradient.get()};

        if (!runtime::finite::all(gradients)) {
            update(false);
            return std::nullopt;
        }

        auto next = detail::create_sgd(parameter, scaled_gradient, step_size);
        update(true);
        return next;
    }

    std::optional<std::vector<std::shared_ptr<Tensor>>> GradScaler::step(
        const std::vector<std::shared_ptr<Tensor>>& parameters,
        const std::vector<std::shared_ptr<Tensor>>& scaled_gradients,
        double learning_rate
    ) {
        if (parameters.empty()) {
            ORMI_THROW(OperationError, "Gradient scaler step requires at least one parameter.");
        }

        if (parameters.size() != scaled_gradients.size()) {
            ORMI_THROW(OperationError, "Gradient scaler parameter and gradient counts must match.");
        }

        const auto step_scale = scale_;
        const auto step_size = detail::check_sgd(
            parameters[0],
            scaled_gradients[0],
            learning_rate,
            step_scale
        );
        std::vector<const Tensor*> gradients;
        gradients.reserve(scaled_gradients.size());
        gradients.push_back(scaled_gradients[0].get());

        for (std::size_t i = 1; i < parameters.size(); ++i) {
            (void)detail::check_sgd(
                parameters[i],
                scaled_gradients[i],
                learning_rate,
                step_scale
            );
            gradients.push_back(scaled_gradients[i].get());
        }

        if (!runtime::finite::all(gradients)) {
            update(false);
            return std::nullopt;
        }

        std::vector<std::shared_ptr<Tensor>> next;
        next.reserve(parameters.size());

        for (std::size_t i = 0; i < parameters.size(); ++i) {
            next.push_back(detail::create_sgd(parameters[i], scaled_gradients[i], step_size));
        }

        update(true);
        return next;
    }

    void GradScaler::update(bool finite) noexcept {
        if (!finite) {
            scale_ = std::max(1.0, scale_ * policy_.backoff);
            finite_steps_ = 0;
            return;
        }

        ++finite_steps_;
        if (finite_steps_ != policy_.interval) {
            return;
        }

        if (scale_ <= policy_.limit / policy_.growth) {
            scale_ *= policy_.growth;
        } else {
            scale_ = policy_.limit;
        }

        finite_steps_ = 0;
    }
}
