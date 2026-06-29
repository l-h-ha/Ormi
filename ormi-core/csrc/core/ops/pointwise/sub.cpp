#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> sub(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::SUB,
                std::vector<std::shared_ptr<Node>>{a, b}
            );
        }
    }

    void adjoint_sub(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];
        const auto& y = node->parents[1];

        if (x->requires_grad) {
            auto local_dx = unbroadcast(dz, x->view.shape);
            ctx.add(x, local_dx);
        }

        if (y->requires_grad) {
            auto neg_dz = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::NEG,
                std::vector<std::shared_ptr<Node>>{dz}
            );

            auto local_dy = unbroadcast(neg_dz, y->view.shape);
            ctx.add(y, local_dy);
        }
    }

    static RegisterShape reg_shape_sub(ops::OpEnum::SUB, infer_binary_shape);
    static RegisterAdjoint reg_adj_sub(ops::OpEnum::SUB, adjoint_sub);
}