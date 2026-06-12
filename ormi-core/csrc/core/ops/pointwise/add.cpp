#include "core/ops.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/adjoint.h"
#include "core/registry.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
            return std::make_shared<Node>(OpEnum::ADD, std::vector<std::shared_ptr<Node>>{a, b}, a->device);
        }
    }

    void adjoint_add(std::shared_ptr<Node> node) {
        auto dz = node->grad;
        for (int i = 0; i < 2; ++i) {
            if (node->parents[i]->requires_grad) {
                auto local_dz = unbroadcast(dz, node->parents[i]->view.shape); 
                if (!node->parents[i]->grad) node->parents[i]->grad = local_dz;
                else node->parents[i]->grad = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{node->parents[i]->grad, local_dz}, node->device);
            }
        }
    }

    static RegisterShape reg_shape_add(ops::OpEnum::ADD, infer_binary_shape);
    static RegisterAdjoint reg_adj_add(ops::OpEnum::ADD, adjoint_add);
}