#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <variant>
#include <vector>

#include <ormi/core/float16.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/nn/modules/batch_norm.h>

#include "nn/functional.h"
#include "nn/state.h"

namespace ormi::nn {
    namespace {
        std::shared_ptr<core::Tensor> state_tensor(
            std::int64_t features,
            core::DType dtype,
            core::Device device,
            bool ones
        ) {
            auto tensor = std::make_shared<core::Tensor>(device, core::Shape{features}, dtype);
            if (!ones) {
                core::memory::zero(*tensor->buffer());
                return tensor;
            }
            core::Scalar one = 1.0f;
            if (dtype == core::DType::float64) {
                one = 1.0;
            } else if (dtype == core::DType::float16) {
                one = core::Float16{1.0f};
            }
            std::visit(
                [&](const auto& value) {
                    core::memory::fill_scalar(tensor->data(), device, &value, tensor->numel(), dtype);
                },
                one
            );
            return tensor;
        }

        void check_state(const core::Tensor* state, const core::Tensor& input, core::DType expected_dtype) {
            if (!state) {
                return;
            }
            if (state->device() != input.device()) {
                ORMI_THROW(DeviceError, "BatchNorm parameters and buffers must use the input device.");
            }

            const auto dtype = state->dtype();
            const bool mixed_half = input.dtype() == core::DType::float16 && dtype == core::DType::float32;
            if (dtype != expected_dtype && !mixed_half) {
                ORMI_THROW(
                    DTypeError,
                    "BatchNorm dtype must match the input. Float16 inputs also accept Float32 state."
                );
            }
        }
    }

    BatchNorm::BatchNorm(
        std::int64_t num_features,
        double eps,
        double momentum,
        bool affine,
        bool track_running_stats,
        std::int64_t axis,
        core::DType dtype,
        core::Device device
    )
        : num_features_(num_features),
          eps_(eps),
          momentum_(momentum),
          axis_(axis) {
        detail::check_normalization(core::Shape{num_features}, eps);
        if (!core::is_floating(dtype)) {
            ORMI_THROW(DTypeError, "BatchNorm requires a floating-point dtype.");
        }
        if (!std::isfinite(momentum) || momentum < 0 || momentum > 1) {
            ORMI_THROW(OperationError, "BatchNorm momentum must be finite and between zero and one.");
        }
        if (dtype != core::DType::float64 &&
            (eps > std::numeric_limits<float>::max() || static_cast<float>(eps) <= 0)) {
            ORMI_THROW(OperationError, "BatchNorm eps must be representable as a positive Float32 value.");
        }
        if (affine) {
            weight_ = state_tensor(num_features, dtype, device, true);
            bias_ = state_tensor(num_features, dtype, device, false);
            register_parameter(weight_, "weight");
            register_parameter(bias_, "bias");
        }
        if (track_running_stats) {
            const auto stats_dtype = dtype == core::DType::float16 ? core::DType::float32 : dtype;
            running_mean_ = state_tensor(num_features, stats_dtype, device, false);
            running_var_ = state_tensor(num_features, stats_dtype, device, true);
            register_buffer(running_mean_, "running_mean");
            register_buffer(running_var_, "running_var");
        }
    }

    std::shared_ptr<core::Tensor> BatchNorm::forward(const std::shared_ptr<core::Tensor>& input) const {
        using namespace core::ops;
        auto value = detail::floating(input);
        auto channel = axis_;
        if (channel < 0) {
            channel += static_cast<std::int64_t>(input->rank());
        }
        if (channel < 0 || channel >= static_cast<std::int64_t>(input->rank())) {
            ORMI_THROW(ShapeError, "BatchNorm channel axis is out of range.");
        }
        if (input->shape()[static_cast<std::size_t>(channel)] != num_features_) {
            ORMI_THROW(ShapeError, "BatchNorm channel size must equal num_features.");
        }
        if (value->dtype() == core::DType::float32 &&
            (eps_ > std::numeric_limits<float>::max() || static_cast<float>(eps_) <= 0)) {
            ORMI_THROW(OperationError, "BatchNorm eps must be representable as a positive Float32 value.");
        }
        check_state(weight_.get(), *input, input->dtype());
        check_state(bias_.get(), *input, input->dtype());
        check_state(running_mean_.get(), *input, value->dtype());
        check_state(running_var_.get(), *input, value->dtype());

        // Two forwards sharing these buffers must update in call order, even if a Plan
        // receives their roots in reverse order. Completed forwards need no dependency.
        if (auto previous = previous_.lock(); previous && !previous->realized()) {
            value = detail::depend(value, previous);
        }
        std::vector<std::int64_t> axes;
        std::vector<std::int64_t> broadcast_dims(input->rank(), 1);
        broadcast_dims[static_cast<std::size_t>(channel)] = num_features_;
        for (std::size_t axis = 0; axis < input->rank(); ++axis) {
            if (axis != static_cast<std::size_t>(channel)) {
                axes.push_back(static_cast<std::int64_t>(axis));
            }
        }
        const core::Shape broadcast_shape{broadcast_dims};
        const auto samples = input->numel() / static_cast<std::size_t>(num_features_);
        if (samples == 0) {
            return cast(mul(value, detail::constant(value, 1)), input->dtype());
        }
        const bool batch_stats = training() || !running_mean_;
        if (batch_stats && samples < 2) {
            ORMI_THROW(ShapeError, "Training BatchNorm requires at least two values per channel.");
        }

        std::shared_ptr<core::Tensor> average;
        std::shared_ptr<core::Tensor> variance;
        std::shared_ptr<core::Tensor> centered;
        if (batch_stats) {
            average = mean(value, axes, true);
            centered = sub(value, average);
            const auto squared = mul(centered, centered);
            variance = mean(squared, axes, true);
        } else {
            average = reshape(running_mean_, broadcast_shape);
            variance = reshape(running_var_, broadcast_shape);
            centered = sub(value, average);
        }
        const auto stabilized = add(variance, detail::constant(value, eps_));
        const auto inverse_stddev = rsqrt(stabilized);
        auto output = mul(centered, inverse_stddev);
        if (weight_) {
            output = mul(output, reshape(cast(weight_, value->dtype()), broadcast_shape));
            output = add(output, reshape(cast(bias_, value->dtype()), broadcast_shape));
        }
        output = cast(output, input->dtype());
        if (!training() || !running_mean_ || momentum_ == 0) {
            return output;
        }

        const auto momentum = detail::constant(value, momentum_);
        const auto retained = detail::constant(value, 1 - momentum_);
        // Normalize with population variance, but store the N/(N-1) corrected running estimate.
        const auto correction = detail::constant(value, static_cast<double>(samples) / (samples - 1));
        const auto batch_mean = reshape(average, core::Shape{num_features_});
        const auto batch_variance = reshape(mul(variance, correction), core::Shape{num_features_});
        const auto next_mean = add(mul(running_mean_, retained), mul(batch_mean, momentum));
        const auto next_variance = add(mul(running_var_, retained), mul(batch_variance, momentum));
        output = detail::update(output, running_mean_, next_mean);
        output = detail::update(output, running_var_, next_variance);
        previous_ = output;
        return output;
    }

    std::int64_t BatchNorm::num_features() const noexcept {
        return num_features_;
    }

    std::int64_t BatchNorm::axis() const noexcept {
        return axis_;
    }

    double BatchNorm::eps() const noexcept {
        return eps_;
    }

    double BatchNorm::momentum() const noexcept {
        return momentum_;
    }

    std::shared_ptr<core::Tensor> BatchNorm::weight() const noexcept {
        return weight_;
    }

    std::shared_ptr<core::Tensor> BatchNorm::bias() const noexcept {
        return bias_;
    }

    std::shared_ptr<core::Tensor> BatchNorm::running_mean() const noexcept {
        return running_mean_;
    }

    std::shared_ptr<core::Tensor> BatchNorm::running_var() const noexcept {
        return running_var_;
    }
}
