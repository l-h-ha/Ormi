#pragma once
#include <memory>
#include <vector>
#include "core/attributes.h"
#include "core/node.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> reshape(const std::shared_ptr<Node>& node, std::vector<int> shape);
    }

    std::vector<int> infer_reshape_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs);
    std::vector<int> infer_reshape_strides(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs, const std::vector<int>& out_shape);
}