#pragma once
#include <vector>
#include <memory>
#include "core/ops.h"
#include "core/attributes.h"
#include "core/registry.h"

namespace ormi {
    struct Node;

    struct TensorView {
        std::vector<int> shape;
        std::vector<int> strides;

        bool is_contiguous() const;
    };

    std::vector<int> calc_contiguous_strides(const std::vector<int>& shape);
    std::vector<int> broadcast_shapes(const std::vector<int>& shape_a, const std::vector<int>& shape_b);
    
    std::vector<int> infer_shape(ops::OpEnum op, const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs);

    // Generic shape rules
    std::vector<int> infer_unary_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes&);
    std::vector<int> infer_binary_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes&);
}