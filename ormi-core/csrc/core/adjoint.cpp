#include "core/adjoint.h"
#include "core/shape.h"

namespace ormi {
    std::shared_ptr<Node> unbroadcast(std::shared_ptr<Node> grad, const std::vector<int>& target_shape) {
        if (grad->view.shape == target_shape) return grad;
    
        std::vector<int> reduce_axes;
        int ndim_grad = grad->view.shape.size();
        int ndim_target = target_shape.size();
    
        for (int i = 0; i < ndim_grad - ndim_target; ++i) reduce_axes.push_back(i);
        for (int i = ndim_grad - ndim_target; i < ndim_grad; ++i) {
            int target_dim = target_shape[i - (ndim_grad - ndim_target)];
            if (target_dim == 1 && grad->view.shape[i] > 1) {
                reduce_axes.push_back(i);
            }
        }
    
        if (reduce_axes.empty()) return grad;

        ReduceAttrs attrs;
        attrs.axes = reduce_axes;
        attrs.keepdims = false; 

        auto sum_node = std::make_shared<Node>(
            ops::OpEnum::SUM, std::vector<std::shared_ptr<Node>>{grad}, grad->device, attrs
        );
        
        sum_node->view.shape = target_shape;
        sum_node->view.strides = calc_contiguous_strides(target_shape);
        
        return sum_node;
    }
}