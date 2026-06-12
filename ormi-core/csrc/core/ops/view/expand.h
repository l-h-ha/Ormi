#pragma once
#include <memory>
#include <vector>
#include "core/attributes.h"
#include "core/node.h"

namespace ormi {
    namespace ops {
        std::shared_ptr<Node> expand(std::shared_ptr<Node> node, const std::vector<int>& shape);
    }

    std::vector<int> infer_expand_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs);
    void adjoint_expand(std::shared_ptr<Node> node);
    std::vector<int> infer_expand_strides(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs, const std::vector<int>& out_shape);
}