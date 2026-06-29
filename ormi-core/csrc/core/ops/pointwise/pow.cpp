#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/node_allocator.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> pow(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            return std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                OpEnum::POW,
                std::vector<std::shared_ptr<Node>>{a, b}
            );
        }
    }

    void adjoint_pow(const std::shared_ptr<Node>& node, GradContext& ctx) {
        const auto& dz = ctx.get(node);

        if (!dz) {
            return;
        }

        const auto& x = node->parents[0];
        const auto& y = node->parents[1];

        if (x->requires_grad) {
            // dX = dZ * y * x^(y-1)
            auto ones = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                full(y->view.shape, 1.0f, y->device, y->dtype),
                false
            );

            auto y_minus_1 = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::SUB,
                std::vector<std::shared_ptr<Node>>{y, ones}
            );

            auto x_pow = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::POW,
                std::vector<std::shared_ptr<Node>>{x, y_minus_1}
            );

            auto y_mul_x_pow = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{y, x_pow}
            );

            auto dz_mul = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{dz, y_mul_x_pow}
            );

            auto local_dx = unbroadcast(dz_mul, x->view.shape);
            ctx.add(x, local_dx);
        }

        if (y->requires_grad) {
            // dY = dZ * x^y * ln(x)
            auto log_x = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::LOG,
                std::vector<std::shared_ptr<Node>>{x}
            );

            auto node_mul_log = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{node, log_x}
            );

            auto dz_mul = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::MUL,
                std::vector<std::shared_ptr<Node>>{dz, node_mul_log}
            );

            auto local_dy = unbroadcast(dz_mul, y->view.shape);
            ctx.add(y, local_dy);
        }
    }

    static RegisterShape reg_shape_pow(ops::OpEnum::POW, infer_binary_shape);
    static RegisterAdjoint reg_adj_pow(ops::OpEnum::POW, adjoint_pow);
}