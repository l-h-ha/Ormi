#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include <ormi/core/error.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/reduction.h>

namespace ormi::nn::detail {
    inline void check_normalization(const core::Shape& normalized_shape, double eps) {
        if (normalized_shape.rank() == 0 || normalized_shape.empty()) {
            ORMI_THROW(ShapeError, "Normalized shape must have at least one axis, all with positive sizes.");
        }
        if (!std::isfinite(eps) || eps <= 0) {
            ORMI_THROW(OperationError, "Normalization eps must be finite and positive.");
        }
    }

    inline std::shared_ptr<core::Tensor> floating(const std::shared_ptr<core::Tensor>& input) {
        if (!input) {
            ORMI_THROW(OperationError, "A tensor is required.");
        }
        if (!core::is_floating(input->dtype())) {
            ORMI_THROW(DTypeError, "NN operations require floating-point tensors.");
        }

        return input->dtype() == core::DType::float16 ? core::ops::cast(input, core::DType::float32) : input;
    }

    inline std::shared_ptr<core::Tensor> constant(const std::shared_ptr<core::Tensor>& input, double value) {
        core::Scalar scalar = value;

        if (input->dtype() == core::DType::float32) {
            scalar = static_cast<float>(value);
        }
        if (input->dtype() == core::DType::float16) {
            scalar = core::Float16(static_cast<float>(value));
        }

        return core::ops::full(core::Shape{}, scalar, input->device());
    }

    inline std::int64_t axis(const core::Tensor& input, std::int64_t value) {
        if (value < 0) {
            value += static_cast<std::int64_t>(input.rank());
        }

        if (value < 0 || value >= static_cast<std::int64_t>(input.rank())) {
            ORMI_THROW(ShapeError, "Class axis is out of range.");
        }
        if (input.shape()[static_cast<std::size_t>(value)] == 0) {
            ORMI_THROW(ShapeError, "Class axis cannot be empty.");
        }

        return value;
    }

    inline std::pair<std::shared_ptr<core::Tensor>, std::shared_ptr<core::Tensor>> loss_inputs(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target
    ) {
        auto values = floating(prediction);
        auto targets = floating(target);

        if (values->shape() != targets->shape()) {
            ORMI_THROW(ShapeError, "Loss prediction and target shapes must match.");
        }
        if (values->device() != targets->device()) {
            ORMI_THROW(DeviceError, "Loss prediction and target must use the same device.");
        }

        const auto dtype = core::promote(values->dtype(), targets->dtype());

        if (values->dtype() != dtype) {
            values = core::ops::cast(values, dtype);
        }
        if (targets->dtype() != dtype) {
            targets = core::ops::cast(targets, dtype);
        }

        return {std::move(values), std::move(targets)};
    }

    inline void check_reduction(std::string_view reduction) {
        if (reduction != "none" && reduction != "sum" && reduction != "mean") {
            ORMI_THROW(OperationError, "Loss reduction must be 'none', 'sum', or 'mean'.");
        }
    }

    inline std::shared_ptr<core::Tensor> reduce(
        const std::shared_ptr<core::Tensor>& value,
        std::string_view reduction
    ) {
        check_reduction(reduction);

        if (reduction == "none") {
            return value;
        }
        if (reduction == "sum") {
            return core::ops::sum(value);
        }

        return core::ops::mean(value);
    }
}
