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

    inline std::ostream& operator<<(std::ostream& os, const TensorView& tv) {
        auto print_vector = [&os](const std::vector<int>& vec) {
            os << "[";
            for (size_t i = 0; i < vec.size(); ++i) {
                os << vec[i];
                if (i != vec.size() - 1) {
                    os << ", ";
                }
            }
            os << "]";
        };

        os << "TensorView(";
        os << "shape=";
        print_vector(tv.shape);

        os << ", strides=";
        print_vector(tv.strides);

        os << ", is_contiguous=" << std::boolalpha << tv.is_contiguous();
        os << ")";
        os << std::noboolalpha;

        return os;
    }

    std::vector<int> calc_contiguous_strides(const std::vector<int>& shape);
    std::vector<int> broadcast_shapes(const std::vector<int>& shape_a, const std::vector<int>& shape_b);
    std::vector<int> infer_shape(ops::OpEnum op, const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs);
    std::vector<int> infer_unary_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes&);
    std::vector<int> infer_binary_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes&);
}