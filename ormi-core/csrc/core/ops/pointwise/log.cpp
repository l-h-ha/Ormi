#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> log(const std::shared_ptr<Node>& a) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::LOG,
                std::vector<std::shared_ptr<Node>>{a}
            );
        }
    }

    void adjoint_log(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];

        if (x->requires_grad) {
            // dX = dZ * (1 / X) -> dZ / X
            auto dz_div_x = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::DIV,
                std::vector<std::shared_ptr<Node>>{dz, x}
            );

            auto local_dx = unbroadcast(dz_div_x, x->view.shape);
            ctx.add(x, local_dx);
        }
    }

    static RegisterShape reg_shape_log(ops::OpEnum::LOG, infer_unary_shape);
    static RegisterAdjoint reg_adj_log(ops::OpEnum::LOG, adjoint_log);
}