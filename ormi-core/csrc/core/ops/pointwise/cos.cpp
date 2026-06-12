#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> cos(std::shared_ptr<Node> a) { return std::make_shared<Node>(OpEnum::COS, std::vector<std::shared_ptr<Node>>{a}, a->device); }
    }
    void adjoint_cos(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];
        if (x->requires_grad) {
            auto sin_x = std::make_shared<Node>(ops::OpEnum::SIN, std::vector<std::shared_ptr<Node>>{x}, node->device);
            auto neg_sin_x = std::make_shared<Node>(ops::OpEnum::NEG, std::vector<std::shared_ptr<Node>>{sin_x}, node->device);
            auto local_dx = unbroadcast(std::make_shared<Node>(ops::OpEnum::MUL, std::vector<std::shared_ptr<Node>>{neg_sin_x, dz}, node->device), x->view.shape);
            if (!x->grad) x->grad = local_dx;
            else x->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, node->device);
        }
    }
    static RegisterShape reg_shape_cos(ops::OpEnum::COS, infer_unary_shape);
    static RegisterAdjoint reg_adj_cos(ops::OpEnum::COS, adjoint_cos);
}