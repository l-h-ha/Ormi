#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
            return std::make_shared<Node>(OpEnum::SUB, std::vector<std::shared_ptr<Node>>{a, b}, a->device);
        }
    }

    void adjoint_sub(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];
        auto y = node->parents[1];

        if (x->requires_grad) {
            auto local_dx = unbroadcast(dz, x->view.shape);
            if (!x->grad) x->grad = local_dx;
            else x->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, node->device);
        }
        if (y->requires_grad) {
            auto neg_dz = std::make_shared<Node>(ops::OpEnum::NEG, std::vector<std::shared_ptr<Node>>{dz}, node->device);
            auto local_dy = unbroadcast(neg_dz, y->view.shape);
            if (!y->grad) y->grad = local_dy;
            else y->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{y->grad, local_dy}, node->device);
        }
    }

    static RegisterShape reg_shape_sub(ops::OpEnum::SUB, infer_binary_shape);
    static RegisterAdjoint reg_adj_sub(ops::OpEnum::SUB, adjoint_sub);
}