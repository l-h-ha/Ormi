#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> log(std::shared_ptr<Node> a) {
            return std::make_shared<Node>(OpEnum::LOG, std::vector<std::shared_ptr<Node>>{a}, a->device);
        }
    }

    void adjoint_log(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        auto x = node->parents[0];

        if (x->requires_grad) {
            // dX = dZ * (1 / X) -> dZ / X
            auto dz_div_x = std::make_shared<Node>(ops::OpEnum::DIV, std::vector<std::shared_ptr<Node>>{dz, x}, node->device);
            
            auto local_dx = unbroadcast(dz_div_x, x->view.shape);
            if (!x->grad) x->grad = local_dx;
            else x->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{x->grad, local_dx}, node->device);
        }
    }

    static RegisterShape reg_shape_log(ops::OpEnum::LOG, infer_unary_shape);
    static RegisterAdjoint reg_adj_log(ops::OpEnum::LOG, adjoint_log);
}