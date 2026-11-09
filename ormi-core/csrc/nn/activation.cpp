#include <cmath>
#include <cstdint>
#include <memory>

#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/nn/activation.h>

#include "nn/functional.h"

namespace ormi::nn {
    using namespace core::ops;

    std::shared_ptr<core::Tensor> relu(const std::shared_ptr<core::Tensor>& input) {
        return core::ops::relu(input);
    }

    std::shared_ptr<core::Tensor> sigmoid(const std::shared_ptr<core::Tensor>& input) {
        return core::ops::sigmoid(input);
    }

    std::shared_ptr<core::Tensor> tanh(const std::shared_ptr<core::Tensor>& input) {
        return core::ops::tanh(input);
    }

    std::shared_ptr<core::Tensor> softplus(const std::shared_ptr<core::Tensor>& input) {
        return core::ops::softplus(input);
    }

    std::shared_ptr<core::Tensor> leaky_relu(
        const std::shared_ptr<core::Tensor>& input,
        double negative_slope
    ) {
        if (!std::isfinite(negative_slope) || negative_slope < 0) {
            ORMI_THROW(OperationError, "LeakyReLU negative_slope must be finite and nonnegative.");
        }

        const auto value = detail::floating(input);
        const auto slope = detail::constant(value, negative_slope);
        const auto linear_part = mul(slope, value);
        const auto relu_scale = sub(detail::constant(value, 1), slope);
        const auto relu_part = mul(relu_scale, core::ops::relu(value));
        const auto output = add(linear_part, relu_part);

        return cast(output, input->dtype());
    }

    std::shared_ptr<core::Tensor> gelu(const std::shared_ptr<core::Tensor>& input) {
        constexpr double inverse_sqrt_two = 0.7071067811865475244;
        const auto value = detail::floating(input);
        const auto scaled_input = mul(value, detail::constant(value, inverse_sqrt_two));
        const auto half_input = mul(detail::constant(value, 0.5), value);
        const auto one_plus_erf = add(detail::constant(value, 1), erf(scaled_input));
        const auto output = mul(half_input, one_plus_erf);

        return cast(output, input->dtype());
    }

    std::shared_ptr<core::Tensor> silu(const std::shared_ptr<core::Tensor>& input) {
        const auto value = detail::floating(input);
        const auto gate = core::ops::sigmoid(value);
        const auto output = mul(value, gate);

        return cast(output, input->dtype());
    }

    std::shared_ptr<core::Tensor> softsign(const std::shared_ptr<core::Tensor>& input) {
        const auto value = detail::floating(input);
        const auto denominator = add(detail::constant(value, 1), abs(value));
        const auto output = div(value, denominator);

        return cast(output, input->dtype());
    }

    std::shared_ptr<core::Tensor> mish(const std::shared_ptr<core::Tensor>& input) {
        const auto value = detail::floating(input);
        const auto softplus_value = core::ops::softplus(value);
        const auto gate = core::ops::tanh(softplus_value);
        const auto output = mul(value, gate);

        return cast(output, input->dtype());
    }

    std::shared_ptr<core::Tensor> softmax(const std::shared_ptr<core::Tensor>& input, std::int64_t axis) {
        const auto value = detail::floating(input);
        axis = detail::axis(*value, axis);

        const auto maximum = max(value, {axis}, true);
        const auto shifted = sub(value, maximum);
        const auto weights = exp(shifted);
        const auto total = sum(weights, {axis}, true);
        const auto output = div(weights, total);

        return cast(output, input->dtype());
    }

    std::shared_ptr<core::Tensor> log_softmax(const std::shared_ptr<core::Tensor>& input, std::int64_t axis) {
        const auto value = detail::floating(input);
        axis = detail::axis(*value, axis);

        const auto maximum = max(value, {axis}, true);
        const auto shifted = sub(value, maximum);
        const auto weights = exp(shifted);
        const auto total = sum(weights, {axis}, true);
        const auto log_total = log(total);
        const auto output = sub(shifted, log_total);

        return cast(output, input->dtype());
    }
}
