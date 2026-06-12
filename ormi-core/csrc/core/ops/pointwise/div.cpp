#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> div(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
            return std::make_shared<Node>(OpEnum::DIV, std::vector<std::shared_ptr<Node>>{a, b}, a->device);
        }
    }

    void adjoint_div(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];
        auto y = node->parents[1];

        if (x->requires_grad) {
            // dX = dZ / Y
            auto local_dx = unbroadcast(std::make_shared<Node>(ops::OpEnum::DIV, std::vector<std::shared_ptr<Node>>{dz, y}, node->device), x->view.shape);
            if (!x->grad) x->grad = local_dx;
            else x->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, node->device);
        }
        if (y->requires_grad) {
            // dY = -dZ * X / (Y * Y)
            auto y_sq = std::make_shared<Node>(ops::OpEnum::MUL, std::vector<std::shared_ptr<Node>>{y, y}, node->device);
            auto x_div_ysq = std::make_shared<Node>(ops::OpEnum::DIV, std::vector<std::shared_ptr<Node>>{x, y_sq}, node->device);
            auto dz_mul = std::make_shared<Node>(ops::OpEnum::MUL, std::vector<std::shared_ptr<Node>>{dz, x_div_ysq}, node->device);
            auto neg_dy = std::make_shared<Node>(ops::OpEnum::NEG, std::vector<std::shared_ptr<Node>>{dz_mul}, node->device);
            
            auto local_dy = unbroadcast(neg_dy, y->view.shape);
            if (!y->grad) y->grad = local_dy;
            else y->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{y->grad, local_dy}, node->device);
        }
    }

    static RegisterShape reg_shape_div(ops::OpEnum::DIV, infer_binary_shape);
    static RegisterAdjoint reg_adj_div(ops::OpEnum::DIV, adjoint_div);
}