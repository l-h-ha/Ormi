#pragma once
#include <memory>
#include <vector>
#include "core/attributes.h"
#include "core/node.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> sum(const std::shared_ptr<Node>& node, std::vector<int> axes, const bool keepdims = false);
        std::shared_ptr<Node> sum(const std::shared_ptr<Node>& node, const bool keepdims = false);
    }

    std::vector<int> infer_sum_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs);
}