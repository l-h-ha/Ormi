#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> pow(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
            return std::make_shared<Node>(OpEnum::POW, std::vector<std::shared_ptr<Node>>{a, b}, a->device);
        }
    }

    void adjoint_pow(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];
        auto y = node->parents[1];

        if (x->requires_grad) {
            // dX = dZ * y * x^(y-1)
            auto ones = std::make_shared<Node>(full(y->view.shape, 1.0f, y->device.type), false, y->device);
            auto y_minus_1 = std::make_shared<Node>(ops::OpEnum::SUB, std::vector<std::shared_ptr<Node>>{y, ones}, node->device);
            auto x_pow = std::make_shared<Node>(ops::OpEnum::POW, std::vector<std::shared_ptr<Node>>{x, y_minus_1}, node->device);
            
            auto y_mul_x_pow = std::make_shared<Node>(ops::OpEnum::MUL, std::vector<std::shared_ptr<Node>>{y, x_pow}, node->device);
            auto dz_mul = std::make_shared<Node>(ops::OpEnum::MUL, std::vector<std::shared_ptr<Node>>{dz, y_mul_x_pow}, node->device);
            
            auto local_dx = unbroadcast(dz_mul, x->view.shape);
            if (!x->grad) x->grad = local_dx;
            else x->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, node->device);
        }

        if (y->requires_grad) {
            // dY = dZ * x^y * ln(x)
            auto log_x = std::make_shared<Node>(ops::OpEnum::LOG, std::vector<std::shared_ptr<Node>>{x}, node->device);
            auto node_mul_log = std::make_shared<Node>(ops::OpEnum::MUL, std::vector<std::shared_ptr<Node>>{node, log_x}, node->device);
            auto dz_mul = std::make_shared<Node>(ops::OpEnum::MUL, std::vector<std::shared_ptr<Node>>{dz, node_mul_log}, node->device);
            
            auto local_dy = unbroadcast(dz_mul, y->view.shape);
            if (!y->grad) y->grad = local_dy;
            else y->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{y->grad, local_dy}, node->device);
        }
    }

    static RegisterShape reg_shape_pow(ops::OpEnum::POW, infer_binary_shape);
    static RegisterAdjoint reg_adj_pow(ops::OpEnum::POW, adjoint_pow);
}