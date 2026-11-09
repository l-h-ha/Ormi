#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <vector>

#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/nn/normalization.h>

#include "nn/functional.h"

namespace ormi::nn {
    std::shared_ptr<core::Tensor> layer_norm(
        const std::shared_ptr<core::Tensor>& input,
        const core::Shape& normalized_shape,
        const std::shared_ptr<core::Tensor>& weight,
        const std::shared_ptr<core::Tensor>& bias,
        double eps
    ) {
        using namespace core::ops;

        detail::check_normalization(normalized_shape, eps);
        const auto value = detail::floating(input);
        if (normalized_shape.rank() > value->rank()) {
            ORMI_THROW(ShapeError, "LayerNorm normalized shape has more axes than the input.");
        }
        if (value->dtype() == core::DType::float32 &&
            (eps > std::numeric_limits<float>::max() || static_cast<float>(eps) <= 0)) {
            ORMI_THROW(OperationError, "LayerNorm eps must be representable as a positive Float32 value.");
        }

        const auto first_axis = value->rank() - normalized_shape.rank();
        std::vector<std::int64_t> axes;
        axes.reserve(normalized_shape.rank());
        for (std::size_t axis = 0; axis < normalized_shape.rank(); ++axis) {
            if (value->shape()[first_axis + axis] != normalized_shape[axis]) {
                ORMI_THROW(ShapeError, "LayerNorm normalized shape must match the input's trailing axes.");
            }
            axes.push_back(static_cast<std::int64_t>(first_axis + axis));
        }

        for (const auto* parameter : {weight.get(), bias.get()}) {
            if (!parameter) {
                continue;
            }
            if (parameter->shape() != normalized_shape) {
                ORMI_THROW(ShapeError, "LayerNorm affine parameter shape must equal normalized_shape.");
            }
            if (parameter->device() != input->device()) {
                ORMI_THROW(DeviceError, "LayerNorm affine parameters must use the input device.");
            }
            const bool same_dtype = parameter->dtype() == input->dtype();
            const bool mixed_half =
                input->dtype() == core::DType::float16 && parameter->dtype() == core::DType::float32;
            if (!same_dtype && !mixed_half) {
                ORMI_THROW(
                    DTypeError,
                    "LayerNorm affine dtype must match the input, or be Float32 for Float16 input."
                );
            }
        }

        const auto average = mean(value, axes, true);
        const auto centered = sub(value, average);
        const auto squared = mul(centered, centered);
        const auto variance = mean(squared, axes, true);
        const auto stabilized = add(variance, detail::constant(value, eps));
        const auto inverse_stddev = rsqrt(stabilized);
        auto output = mul(centered, inverse_stddev);

        if (weight) {
            output = mul(output, cast(weight, value->dtype()));
        }
        if (bias) {
            output = add(output, cast(bias, value->dtype()));
        }

        return cast(output, input->dtype());
    }
}
