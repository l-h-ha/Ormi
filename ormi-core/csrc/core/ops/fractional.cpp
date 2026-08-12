#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <ormi/core/attrs.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/fractional.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/ops/fractional/nodes.h"
#include "core/ops/fractional/weights.h"

namespace ormi::core::ops {
    namespace {
        Shape without_axis(const Shape& shape, std::size_t axis) {
            auto dims = shape.dims();
            dims.erase(dims.begin() + static_cast<std::ptrdiff_t>(axis));
            return Shape(std::move(dims));
        }

        bool same_device(const Tensor& lhs, const Tensor& rhs) {
            return
                lhs.device().backend_id() == rhs.device().backend_id() &&
                lhs.device().index() == rhs.device().index();
        }
    }

    std::shared_ptr<Tensor> caputo(
        const std::shared_ptr<Tensor>& input,
        double alpha,
        double spacing,
        std::int64_t axis,
        const std::shared_ptr<Tensor>& initial_derivative
    ) {
        if (!input) {
            ORMI_THROW(
                OperationError,
                "Caputo differentiation requires a tensor."
            );
        }
        if (input->rank() == 0) {
            ORMI_THROW(
                ShapeError,
                "Caputo differentiation requires at least one axis."
            );
        }
        if (
            !std::isfinite(alpha) ||
            alpha <= 0.0 ||
            alpha >= 2.0 ||
            alpha == 1.0
        ) {
            ORMI_THROW(
                OperationError,
                "Caputo alpha must be finite, between zero and two, and not one."
            );
        }
        if (!std::isfinite(spacing) || spacing <= 0.0) {
            ORMI_THROW(
                OperationError,
                "Caputo spacing must be finite and positive."
            );
        }
        if (
            input->dtype() != DType::float32 &&
            input->dtype() != DType::float64
        ) {
            ORMI_THROW(
                DTypeError,
                "Caputo differentiation supports Float32 and Float64."
            );
        }

        if (axis < 0) {
            axis += static_cast<std::int64_t>(input->rank());
        }
        if (axis < 0 || axis >= static_cast<std::int64_t>(input->rank())) {
            ORMI_THROW(ShapeError, "Caputo axis is out of range.");
        }

        const auto high_order = alpha > 1.0;
        if (!high_order && initial_derivative) {
            ORMI_THROW(
                OperationError,
                "An initial derivative is only valid when Caputo alpha is above one."
            );
        }
        if (high_order && !initial_derivative) {
            ORMI_THROW(
                OperationError,
                "Caputo alpha above one requires the initial first derivative."
            );
        }

        const auto axis_index = static_cast<std::size_t>(axis);
        if (initial_derivative) {
            if (initial_derivative->shape() != without_axis(input->shape(), axis_index)) {
                ORMI_THROW(
                    ShapeError,
                    "The initial derivative shape must equal the input shape with the Caputo axis removed."
                );
            }
            if (initial_derivative->dtype() != input->dtype()) {
                ORMI_THROW(
                    DTypeError,
                    "The initial derivative dtype must match the input dtype."
                );
            }
            if (!same_device(*initial_derivative, *input)) {
                ORMI_THROW(
                    DeviceError,
                    "The initial derivative device must match the input device."
                );
            }
        }

        const auto count = static_cast<std::size_t>(input->shape()[axis_index]);
        auto values = fractional::weights(count, alpha, spacing);
        auto coefficients = std::make_shared<Caputo::Coefficients>(
            Caputo::Coefficients{
                .forward = values,
                .adjoint = fractional::adjoint_weights(values, alpha)
            }
        );
        Caputo attrs{
            .alpha = alpha,
            .spacing = spacing,
            .axis = axis,
            .count = count,
            .coefficients = std::move(coefficients)
        };

        if (high_order) {
            return fractional::l2(
                input,
                initial_derivative,
                std::move(attrs)
            );
        }
        return fractional::l1(input, std::move(attrs));
    }

    bool is_caputo(const Tensor& tensor) {
        return fractional::is_node(tensor);
    }
}
