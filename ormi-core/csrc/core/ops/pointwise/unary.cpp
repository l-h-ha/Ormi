#include <memory>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/pointwise.h"
#include "core/precision.h"

namespace ormi::core::ops {
    namespace {

        void compute_neg(Tensor& out) {
            runtime::pointwise::unary(core::ops::UnaryOp::neg, out, *out.parents()[0]);
        }

        void compute_exp(Tensor& out) {
            runtime::pointwise::unary(core::ops::UnaryOp::exp, out, *out.parents()[0]);
        }

        void compute_log(Tensor& out) {
            runtime::pointwise::unary(core::ops::UnaryOp::log, out, *out.parents()[0]);
        }

        void compute_sin(Tensor& out) {
            runtime::pointwise::unary(core::ops::UnaryOp::sin, out, *out.parents()[0]);
        }

        void compute_cos(Tensor& out) {
            runtime::pointwise::unary(core::ops::UnaryOp::cos, out, *out.parents()[0]);
        }

        void adjoint_neg(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], neg(upstream));
        }

        void adjoint_exp(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], mul(upstream, node));
        }

        void adjoint_log(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], div(upstream, node->parents()[0]));
        }

        void adjoint_sin(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], mul(upstream, cos(node->parents()[0])));
        }

        void adjoint_cos(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], neg(mul(upstream, sin(node->parents()[0]))));
        }

        const Op neg_op{
            .name = "neg",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_neg,
            .adjoint = adjoint_neg,
            .scalar = UnaryOp::neg
        };
        const Op exp_op{
            .name = "exp",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_exp,
            .adjoint = adjoint_exp,
            .scalar = UnaryOp::exp,
            .precision = Precision::float32
        };
        const Op log_op{
            .name = "log",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_log,
            .adjoint = adjoint_log,
            .scalar = UnaryOp::log,
            .precision = Precision::float32
        };
        const Op sin_op{
            .name = "sin",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_sin,
            .adjoint = adjoint_sin,
            .scalar = UnaryOp::sin,
            .precision = Precision::float32
        };
        const Op cos_op{
            .name = "cos",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_cos,
            .adjoint = adjoint_cos,
            .scalar = UnaryOp::cos,
            .precision = Precision::float32
        };

        std::shared_ptr<Tensor> unary(const Op& op, const std::shared_ptr<Tensor>& input) {
            if (!input) {
                ORMI_THROW(OperationError, "Pointwise unary operations require a tensor.");
            }

            if (&op == &neg_op) {
                if (!is_numeric(input->dtype())) {
                    ORMI_THROW(DTypeError, "Negation requires a numeric tensor.");
                }

                return std::make_shared<Tensor>(
                    op,
                    std::vector<std::shared_ptr<Tensor>>{input},
                    input->device(),
                    input->shape(),
                    input->dtype()
                );
            }

            if (!is_floating(input->dtype())) {
                ORMI_THROW(DTypeError, "Transcendental operations require floating-point tensors.");
            }

            if (!precision::state.enabled) {
                return std::make_shared<Tensor>(
                    op,
                    std::vector<std::shared_ptr<Tensor>>{input},
                    input->device(),
                    input->shape(),
                    input->dtype()
                );
            }

            const auto dtype = precision::apply(op.precision, input->dtype());
            const auto parent = input->dtype() == dtype ? input : cast(input, dtype);

            return std::make_shared<Tensor>(
                op,
                std::vector<std::shared_ptr<Tensor>>{parent},
                input->device(),
                input->shape(),
                dtype
            );
        }
    }

    std::shared_ptr<Tensor> neg(const std::shared_ptr<Tensor>& input) {
        return unary(neg_op, input);
    }

    std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& input) {
        return unary(exp_op, input);
    }

    std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& input) {
        return unary(log_op, input);
    }

    std::shared_ptr<Tensor> sin(const std::shared_ptr<Tensor>& input) {
        return unary(sin_op, input);
    }

    std::shared_ptr<Tensor> cos(const std::shared_ptr<Tensor>& input) {
        return unary(cos_op, input);
    }
}
