#pragma once
#include <memory>
#include <vector>
#include "core/attributes.h"
#include "core/node.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> matmul(std::shared_ptr<Node> A, std::shared_ptr<Node> B);
    }

    std::vector<int> infer_matmul_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs);
    void adjoint_matmul(std::shared_ptr<Node> node);
}