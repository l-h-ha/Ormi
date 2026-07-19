#include <cstdint>
#include <memory>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core {
    std::shared_ptr<Tensor> unbroadcast(
        const std::shared_ptr<Tensor>& gradient,
        const Shape& shape
    ) {
        if (!gradient) {
            ORMI_THROW(OperationError, "Unbroadcast requires a gradient tensor.");
        }

        if (shape == gradient->shape()) {
            return gradient;
        }

        if (shape.rank() > gradient->rank()) {
            ORMI_THROW(ShapeError, "Gradient rank cannot be smaller than its target rank.");
        }

        const auto rank_offset = gradient->rank() - shape.rank();
        std::vector<std::int64_t> axes;
        axes.reserve(gradient->rank());

        for (std::size_t axis = 0; axis < rank_offset; ++axis) {
            axes.push_back(static_cast<std::int64_t>(axis));
        }

        for (std::size_t axis = 0; axis < shape.rank(); ++axis) {
            const auto gradient_axis = axis + rank_offset;
            const auto target_dim = shape[axis];
            const auto gradient_dim = gradient->shape()[gradient_axis];

            if (target_dim == gradient_dim) {
                continue;
            }

            if (target_dim != 1) {
                ORMI_THROW(ShapeError, "Gradient cannot be unbroadcast to the target shape.");
            }

            axes.push_back(static_cast<std::int64_t>(gradient_axis));
        }

        auto result = axes.empty() ? gradient : ops::sum(gradient, std::move(axes), true);
        return result->shape() == shape ? result : ops::reshape(result, shape);
    }
}
