#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> cos(const std::shared_ptr<Node>& a) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::COS,
                std::vector<std::shared_ptr<Node>>{a}
            );
        }
    }

    void adjoint_cos(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];

        if (x->requires_grad) {
            auto sin_x = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::SIN,
                std::vector<std::shared_ptr<Node>>{x}
            );

            auto neg_sin_x = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::NEG,
                std::vector<std::shared_ptr<Node>>{sin_x}
            );

            auto local_dx = unbroadcast(
                std::allocate_shared<Node>(
                    get_node_allocator<Node>(),
                    ops::OpEnum::MUL,
                    std::vector<std::shared_ptr<Node>>{neg_sin_x, dz}
                ),
                x->view.shape
            );

            ctx.add(x, local_dx);
        }
    }

    static RegisterShape reg_shape_cos(ops::OpEnum::COS, infer_unary_shape);
    static RegisterAdjoint reg_adj_cos(ops::OpEnum::COS, adjoint_cos);
}