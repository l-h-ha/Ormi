#include <cmath>
#include <memory>
#include <vector>

#include <ormi/core/attrs.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/optimizer.h"
#include "core/optim/sgd_detail.h"

namespace ormi::core::optim {
    namespace {
        void compute_sgd(Tensor& out) {
            runtime::optimizer::sgd(
                out,
                *out.parents()[0],
                *out.parents()[1],
                std::get<SGD>(out.attrs()).step_size
            );
        }

        void adjoint_sgd(
            const std::shared_ptr<Tensor>&,
            const std::shared_ptr<Tensor>&,
            Grads&
        ) {}

        const Op sgd_op{
            .name = "sgd",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_sgd,
            .adjoint = adjoint_sgd,
            .scalar = std::monostate{}
        };
    }

    namespace detail {
        double check_sgd(
            const std::shared_ptr<Tensor>& parameter,
            const std::shared_ptr<Tensor>& gradient,
            double learning_rate,
            double gradient_scale
        ) {
            if (!parameter || !gradient) {
                ORMI_THROW(OperationError, "SGD requires a parameter and gradient.");
            }

            if (!std::isfinite(learning_rate) || learning_rate < 0.0) {
                ORMI_THROW(OperationError, "SGD learning rate must be finite and non-negative.");
            }

            if (!std::isfinite(gradient_scale) || gradient_scale <= 0.0) {
                ORMI_THROW(OperationError, "SGD gradient scale must be finite and positive.");
            }

            if (parameter->shape() != gradient->shape()) {
                ORMI_THROW(ShapeError, "SGD parameter and gradient shapes must match.");
            }

            if (parameter->dtype() != gradient->dtype()) {
                ORMI_THROW(DTypeError, "SGD parameter and gradient dtypes must match.");
            }

            const auto parameter_device = parameter->device();
            const auto gradient_device = gradient->device();
            if (
                parameter_device.backend_id() != gradient_device.backend_id() ||
                parameter_device.index() != gradient_device.index()
            ) {
                ORMI_THROW(DeviceError, "SGD parameter and gradient devices must match.");
            }

            if (!is_floating(parameter->dtype())) {
                ORMI_THROW(DTypeError, "SGD supports floating-point tensors only.");
            }

            if (learning_rate == 0.0) {
                return 0.0;
            }

            const auto step_size = learning_rate / gradient_scale;
            if (!std::isfinite(step_size) || step_size <= 0.0) {
                ORMI_THROW(OperationError, "SGD effective step size must be finite and positive.");
            }

            return step_size;
        }

        std::shared_ptr<Tensor> create_sgd(
            const std::shared_ptr<Tensor>& parameter,
            const std::shared_ptr<Tensor>& gradient,
            double step_size
        ) {
            if (step_size == 0.0) {
                return parameter;
            }

            return std::make_shared<Tensor>(
                op_ref<sgd_op>(),
                SGD{step_size},
                std::vector<std::shared_ptr<Tensor>>{parameter, gradient},
                parameter->device(),
                parameter->shape(),
                parameter->dtype()
            );
        }
    }

    std::shared_ptr<Tensor> sgd(
        const std::shared_ptr<Tensor>& parameter,
        const std::shared_ptr<Tensor>& gradient,
        double learning_rate,
        double gradient_scale
    ) {
        const auto step_size = detail::check_sgd(
            parameter,
            gradient,
            learning_rate,
            gradient_scale
        );
        return detail::create_sgd(parameter, gradient, step_size);
    }
}
