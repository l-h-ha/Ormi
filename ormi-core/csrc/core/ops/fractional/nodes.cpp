#include <memory>
#include <utility>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/attrs.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/ops/fractional/nodes.h"
#include "core/runtime/dispatch/fractional.h"

namespace ormi::core::ops::fractional {
    namespace {
        extern const Op l1_op;
        extern const Op l2_op;
        extern const Op history_op;
        extern const Op adjoint_op;
        extern const Op initial_op;
        extern const Op initial_adjoint_op;

        std::shared_ptr<Tensor> unary(
            const std::shared_ptr<Tensor>& input,
            Caputo attrs,
            OpRef op,
            Shape shape
        ) {
            return std::make_shared<Tensor>(
                op,
                std::move(attrs),
                std::vector<std::shared_ptr<Tensor>>{input},
                input->device(),
                std::move(shape),
                input->dtype()
            );
        }

        void compute_l1(Tensor& out) {
            runtime::fractional::apply(
                FractionalOp::caputo,
                out,
                *out.parents()[0],
                nullptr,
                std::get<Caputo>(out.attrs())
            );
        }

        void compute_l2(Tensor& out) {
            runtime::fractional::apply(
                FractionalOp::caputo,
                out,
                *out.parents()[0],
                out.parents()[1].get(),
                std::get<Caputo>(out.attrs())
            );
        }

        void compute_adjoint(Tensor& out) {
            runtime::fractional::apply(
                FractionalOp::caputo_adjoint,
                out,
                *out.parents()[0],
                nullptr,
                std::get<Caputo>(out.attrs())
            );
        }

        void compute_initial(Tensor& out) {
            runtime::fractional::apply(
                FractionalOp::caputo_initial,
                out,
                *out.parents()[0],
                nullptr,
                std::get<Caputo>(out.attrs())
            );
        }

        void compute_initial_adjoint(Tensor& out) {
            runtime::fractional::apply(
                FractionalOp::caputo_initial_adjoint,
                out,
                *out.parents()[0],
                nullptr,
                std::get<Caputo>(out.attrs())
            );
        }

        Setup prepare_forward(Tensor& out) {
            return runtime::fractional::prepare(
                FractionalOp::caputo,
                out,
                std::get<Caputo>(out.attrs())
            );
        }

        Setup prepare_adjoint(Tensor& out) {
            return runtime::fractional::prepare(
                FractionalOp::caputo_adjoint,
                out,
                std::get<Caputo>(out.attrs())
            );
        }

        Setup prepare_initial(Tensor& out) {
            return runtime::fractional::prepare(
                FractionalOp::caputo_initial,
                out,
                std::get<Caputo>(out.attrs())
            );
        }

        Setup prepare_initial_adjoint(Tensor& out) {
            return runtime::fractional::prepare(
                FractionalOp::caputo_initial_adjoint,
                out,
                std::get<Caputo>(out.attrs())
            );
        }

        void add_history_adjoint(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(
                node->parents()[0],
                unary(
                    upstream,
                    std::get<Caputo>(node->attrs()),
                    op_ref<adjoint_op>(),
                    node->parents()[0]->shape()
                )
            );
        }

        void adjoint_l2(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            add_history_adjoint(node, upstream, gradients);
            gradients.add(
                node->parents()[1],
                unary(
                    upstream,
                    std::get<Caputo>(node->attrs()),
                    op_ref<initial_adjoint_op>(),
                    node->parents()[1]->shape()
                )
            );
        }

        void adjoint_adjoint(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(
                node->parents()[0],
                unary(
                    upstream,
                    std::get<Caputo>(node->attrs()),
                    op_ref<history_op>(),
                    node->parents()[0]->shape()
                )
            );
        }

        void adjoint_initial(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(
                node->parents()[0],
                unary(
                    upstream,
                    std::get<Caputo>(node->attrs()),
                    op_ref<initial_adjoint_op>(),
                    node->parents()[0]->shape()
                )
            );
        }

        void adjoint_initial_adjoint(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(
                node->parents()[0],
                unary(
                    upstream,
                    std::get<Caputo>(node->attrs()),
                    op_ref<initial_op>(),
                    node->parents()[0]->shape()
                )
            );
        }

        const Op l1_op{
            .name = "caputo",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_l1,
            .prepare = prepare_forward,
            .adjoint = add_history_adjoint,
            .scalar = std::monostate{}
        };

        const Op l2_op{
            .name = "caputo",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_l2,
            .prepare = prepare_forward,
            .adjoint = adjoint_l2,
            .scalar = std::monostate{}
        };

        const Op history_op{
            .name = "caputo_history",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_l1,
            .prepare = prepare_forward,
            .adjoint = add_history_adjoint,
            .scalar = std::monostate{}
        };

        const Op adjoint_op{
            .name = "caputo_adjoint",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_adjoint,
            .prepare = prepare_adjoint,
            .adjoint = adjoint_adjoint,
            .scalar = std::monostate{}
        };

        const Op initial_op{
            .name = "caputo_initial",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_initial,
            .prepare = prepare_initial,
            .adjoint = adjoint_initial,
            .scalar = std::monostate{}
        };

        const Op initial_adjoint_op{
            .name = "caputo_initial_adjoint",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_initial_adjoint,
            .prepare = prepare_initial_adjoint,
            .adjoint = adjoint_initial_adjoint,
            .scalar = std::monostate{}
        };
    }

    std::shared_ptr<Tensor> l1(
        const std::shared_ptr<Tensor>& input,
        Caputo attrs
    ) {
        return unary(input, std::move(attrs), op_ref<l1_op>(), input->shape());
    }

    std::shared_ptr<Tensor> l2(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& initial_derivative,
        Caputo attrs
    ) {
        return std::make_shared<Tensor>(
            op_ref<l2_op>(),
            std::move(attrs),
            std::vector<std::shared_ptr<Tensor>>{
                input,
                initial_derivative
            },
            input->device(),
            input->shape(),
            input->dtype()
        );
    }

    bool is_node(const Tensor& tensor) {
        return tensor.op() == &l1_op || tensor.op() == &l2_op;
    }
}
