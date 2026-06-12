#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> exp(std::shared_ptr<Node> a) { return std::make_shared<Node>(OpEnum::EXP, std::vector<std::shared_ptr<Node>>{a}, a->device); }
    }
    void adjoint_exp(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];
        if (x->requires_grad) {
            auto local_dx = unbroadcast(std::make_shared<Node>(ops::OpEnum::MUL, std::vector<std::shared_ptr<Node>>{node, dz}, node->device), x->view.shape);
            if (!x->grad) x->grad = local_dx;
            else x->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, node->device);
        }
    }
    static RegisterShape reg_shape_exp(ops::OpEnum::EXP, infer_unary_shape);
    static RegisterAdjoint reg_adj_exp(ops::OpEnum::EXP, adjoint_exp);
}