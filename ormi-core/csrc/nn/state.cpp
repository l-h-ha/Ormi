#include <memory>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/tensor.h>

#include "core/runtime/device.h"
#include "nn/state.h"

namespace ormi::nn::detail {
    namespace {
        void copy_state(core::Tensor& output) {
            const auto& destination = output.parents()[1];
            const auto& replacement = output.parents()[2];

            // Do not wait on the host because this copy also runs during CUDA capture.
            core::runtime::backend(destination->device())
                .memory.copy_async(
                    destination->device(),
                    destination->data(),
                    replacement->data(),
                    destination->bytes()
                );
        }

        void depend_adjoint(
            const std::shared_ptr<core::Tensor>& node,
            const std::shared_ptr<core::Tensor>& upstream,
            core::Grads& gradients
        ) {
            // Keep the ordering dependency in higher derivatives without differentiating it.
            gradients.add(node->parents()[0], depend(upstream, node->parents()[1]));
        }

        void update_adjoint(
            const std::shared_ptr<core::Tensor>& node,
            const std::shared_ptr<core::Tensor>& upstream,
            core::Grads& gradients
        ) {
            // A gradient-only Plan must still execute the forward state update.
            gradients.add(node->parents()[0], depend(upstream, node));
        }

        constexpr core::Op dependency{
            .name = "depend",
            .output = core::Output::alias,
            .parents = 2,
            .compute = nullptr,
            .adjoint = depend_adjoint,
            .scalar = {}
        };
        constexpr core::Op assignment{
            .name = "update",
            .output = core::Output::alias,
            .parents = 3,
            .compute = copy_state,
            .adjoint = update_adjoint,
            .scalar = {}
        };
    }

    std::shared_ptr<core::Tensor> depend(
        const std::shared_ptr<core::Tensor>& value,
        const std::shared_ptr<core::Tensor>& after
    ) {
        return std::make_shared<core::Tensor>(core::op_ref<dependency>(), std::vector{value, after});
    }

    std::shared_ptr<core::Tensor> update(
        const std::shared_ptr<core::Tensor>& value,
        const std::shared_ptr<core::Tensor>& destination,
        const std::shared_ptr<core::Tensor>& replacement
    ) {
        if (!destination || !replacement || destination->op() || !destination->realized()) {
            ORMI_THROW(OperationError, "State updates require a realized leaf destination and a value.");
        }
        if (destination->shape() != replacement->shape()) {
            ORMI_THROW(ShapeError, "State update shapes must match.");
        }
        if (destination->dtype() != replacement->dtype()) {
            ORMI_THROW(DTypeError, "State update dtypes must match.");
        }
        if (!destination->is_contiguous() || !replacement->is_contiguous()) {
            ORMI_THROW(OperationError, "State updates require contiguous storage.");
        }

        return std::make_shared<core::Tensor>(
            core::op_ref<assignment>(),
            std::vector{value, destination, replacement}
        );
    }
}
