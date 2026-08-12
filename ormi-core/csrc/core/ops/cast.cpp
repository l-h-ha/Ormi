#include <memory>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/cast.h"

namespace ormi::core::ops {
    namespace {
        void compute_cast(Tensor& out) {
            runtime::cast::run(out, *out.parents()[0]);
        }

        void adjoint_cast(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& parent = node->parents()[0];

            if (!is_floating(node->dtype()) || !is_floating(parent->dtype())) {
                ORMI_THROW(DTypeError, "Autograd cannot differentiate through an integer or Boolean cast.");
            }

            gradients.add(parent, cast(upstream, parent->dtype()));
        }

        const Op cast_op{
            .name = "cast",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_cast,
            .adjoint = adjoint_cast,
            .scalar = std::monostate{}
        };
    }

    std::shared_ptr<Tensor> cast(const std::shared_ptr<Tensor>& input, DType dtype) {
        if (!input) {
            ORMI_THROW(OperationError, "Cast requires a tensor.");
        }

        if (input->dtype() == dtype) {
            return input;
        }

        return std::make_shared<Tensor>(
            cast_op,
            std::vector<std::shared_ptr<Tensor>>{input},
            input->device(),
            input->shape(),
            dtype
        );
    }

    bool is_cast(const Tensor& tensor) {
        return tensor.op() == &cast_op;
    }
}
