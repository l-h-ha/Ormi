#pragma once
#include <memory>
#include <vector>
#include "core/attributes.h"
#include "core/node.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> sum(std::shared_ptr<Node> node, const std::vector<int>& axes, const bool keepdims);
    }

    std::vector<int> infer_sum_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs);
    void adjoint_sum(std::shared_ptr<Node> node);
}