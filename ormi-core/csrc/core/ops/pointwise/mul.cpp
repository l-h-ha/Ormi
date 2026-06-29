#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> mul(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{a, b}
            );
        }
    }

    void adjoint_mul(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];
        const auto& y = node->parents[1];

        if (x->requires_grad) {
            auto dz_mul_y = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{dz, y}
            );

            auto local_dx = unbroadcast(dz_mul_y, x->view.shape);
            ctx.add(x, local_dx);
        }

        if (y->requires_grad) {
            auto dz_mul_x = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{dz, x}
            );

            auto local_dy = unbroadcast(dz_mul_x, y->view.shape);
            ctx.add(y, local_dy);
        }
    }

    static RegisterShape reg_shape_mul(ops::OpEnum::MUL, infer_binary_shape);
    static RegisterAdjoint reg_adj_mul(ops::OpEnum::MUL, adjoint_mul);
}