#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> exp(const std::shared_ptr<Node>& a) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::EXP,
                std::vector<std::shared_ptr<Node>>{a}
            );
        }
    }

    void adjoint_exp(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];

        if (x->requires_grad) {
            auto local_dx = unbroadcast(
                std::allocate_shared<Node>(
                    get_node_allocator<Node>(),
                    ops::OpEnum::MUL,
                    std::vector<std::shared_ptr<Node>>{node, dz}
                ),
                x->view.shape
            );

            ctx.add(x, local_dx);
        }
    }

    static RegisterShape reg_shape_exp(ops::OpEnum::EXP, infer_unary_shape);
    static RegisterAdjoint reg_adj_exp(ops::OpEnum::EXP, adjoint_exp);
}