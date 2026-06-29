#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> sin(const std::shared_ptr<Node>& a) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::SIN,
                std::vector<std::shared_ptr<Node>>{a}
            );
        }
    }

    void adjoint_sin(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];

        if (x->requires_grad) {
            auto cos_x = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::COS,
                std::vector<std::shared_ptr<Node>>{x}
            );

            auto local_dx = unbroadcast(
                std::allocate_shared<Node>(
                    get_node_allocator<Node>(),
                    ops::OpEnum::MUL,
                    std::vector<std::shared_ptr<Node>>{cos_x, dz}
                ),
                x->view.shape
            );

            ctx.add(x, local_dx);
        }
    }
    static RegisterShape reg_shape_sin(ops::OpEnum::SIN, infer_unary_shape);
    static RegisterAdjoint reg_adj_sin(ops::OpEnum::SIN, adjoint_sin);
}