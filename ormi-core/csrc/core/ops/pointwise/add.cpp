#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> add(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::ADD,
                std::vector<std::shared_ptr<Node>>{a, b}
            );
        }
    }

    void adjoint_add(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        for (int i = 0; i < 2; ++i) {
            const auto& parent = node->parents[i];

            if (parent->requires_grad) {
                auto local_dz = unbroadcast(dz, parent->view.shape);
                ctx.add(parent, local_dz);
            }
        }
    }

    static RegisterShape reg_shape_add(ops::OpEnum::ADD, infer_binary_shape);
    static RegisterAdjoint reg_adj_add(ops::OpEnum::ADD, adjoint_add);
}