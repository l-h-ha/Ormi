#include <memory>

#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/nn/losses/mse.h>

#include "nn/functional.h"

namespace ormi::nn {
    std::shared_ptr<core::Tensor> mse_loss(
        const std::shared_ptr<core::Tensor>& prediction,
        const std::shared_ptr<core::Tensor>& target,
        std::string_view reduction
    ) {
        if (!prediction || !target) {
            ORMI_THROW(OperationError, "MSE loss requires a prediction and target tensor.");
        }

        if (prediction->shape() != target->shape()) {
            ORMI_THROW(ShapeError, "MSE loss prediction and target shapes must match.");
        }

        if (!core::is_floating(prediction->dtype()) || !core::is_floating(target->dtype())) {
            ORMI_THROW(DTypeError, "MSE loss requires floating-point tensors.");
        }

        const auto [values, targets] = detail::loss_inputs(prediction, target);
        const auto error = core::ops::sub(values, targets);
        const auto squared_error = core::ops::mul(error, error);
        return detail::reduce(squared_error, reduction);
    }
}
