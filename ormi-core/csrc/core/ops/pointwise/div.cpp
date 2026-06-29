#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> div(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::DIV,
                std::vector<std::shared_ptr<Node>>{a, b}
            );
        }
    }

    void adjoint_div(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];
        const auto& y = node->parents[1];

        if (x->requires_grad) {
            // dX = dZ / Y
            auto local_dx = unbroadcast(
                std::allocate_shared<Node>(
                    get_node_allocator<Node>(),
                    ops::OpEnum::DIV,
                    std::vector<std::shared_ptr<Node>>{dz, y}
                ),
                x->view.shape
            );

            ctx.add(x, local_dx);
        }

        if (y->requires_grad) {
            // dY = -dZ * X / (Y * Y)
            auto y_sq = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{y, y}
            );

            auto x_div_ysq = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::DIV,
                std::vector<std::shared_ptr<Node>>{x, y_sq}
            );

            auto dz_mul = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{dz, x_div_ysq}
            );

            auto neg_dy = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::NEG,
                std::vector<std::shared_ptr<Node>>{dz_mul}
            );

            auto local_dy = unbroadcast(neg_dy, y->view.shape);
            ctx.add(y, local_dy);
        }
    }

    static RegisterShape reg_shape_div(ops::OpEnum::DIV, infer_binary_shape);
    static RegisterAdjoint reg_adj_div(ops::OpEnum::DIV, adjoint_div);
}