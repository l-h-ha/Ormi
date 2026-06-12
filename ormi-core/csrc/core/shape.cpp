#include <stdexcept>
#include <algorithm>

#include "core/shape.h"
#include "core/node.h"
#include "core/attributes.h"
#include "core/registry.h"

namespace ormi {
    bool TensorView::is_contiguous() const {
        int expected_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape[i];
        }
        return true;
    }

    std::vector<int> calc_contiguous_strides(const std::vector<int>& shape) {
        std::vector<int> strides(shape.size(), 1);
        for (int i = (int)shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i+1] * shape[i+1];
        }
        return strides;
    }

    std::vector<int> broadcast_shapes(const std::vector<int>& shape_a, const std::vector<int>& shape_b) {
        std::vector<int> out_shape;
        int ndim_a = shape_a.size();
        int ndim_b = shape_b.size();
        int out_ndim = std::max(ndim_a, ndim_b);
        out_shape.resize(out_ndim);
    
        for (int i = 0; i < out_ndim; ++i) {
            int dim_a = (i < out_ndim - ndim_a) ? 1 : shape_a[i - (out_ndim - ndim_a)];
            int dim_b = (i < out_ndim - ndim_b) ? 1 : shape_b[i - (out_ndim - ndim_b)];
    
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                throw std::runtime_error("Operands could not be broadcast together.");
            }
            out_shape[i] = std::max(dim_a, dim_b);
        }
        return out_shape;
    }
    
    std::vector<int> infer_shape(ops::OpEnum op, const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes& attrs) {
        if (op == ops::OpEnum::LEAF || op == ops::OpEnum::CONST || parents.empty()) {
            return {};
        }
        auto func = Registry<ShapeInferFunc>::get(op);
        if (!func) throw std::runtime_error("Shape inference rule not registered for this OpEnum.");

        return func(parents, attrs);
    }

    // Generic shape inference
    std::vector<int> infer_unary_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes&) {
        return parents[0]->view.shape;
    }
    
    std::vector<int> infer_binary_shape(const std::vector<std::shared_ptr<Node>>& parents, const OpAttributes&) {
        return broadcast_shapes(parents[0]->view.shape, parents[1]->view.shape);
    }
}