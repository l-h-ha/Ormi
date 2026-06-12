#include "core/node.h"
#include "backend/runtime.h"
#include "core/shape.h"
#include "core/registry.h"

namespace ormi {
    // Leaf node constructor
    Node::Node(std::shared_ptr<Buffer> mem, bool req_grad, Device dev) 
        : op(ops::OpEnum::LEAF), data(mem), requires_grad(req_grad), device(dev) {
        
        this->view.shape = mem->shape;
        this->view.strides = calc_contiguous_strides(mem->shape);
    }
    
    // Internal node constructor
    Node::Node(ops::OpEnum operation, std::vector<std::shared_ptr<Node>> parent_nodes, Device dev, OpAttributes attributes)
        : op(operation), parents(parent_nodes), requires_grad(false), device(dev), attrs(attributes) {

        // Inherit requires_grad if any parent requires it
        for (const auto& p : parents) {
            if (p->requires_grad) requires_grad = true;
        }

        this->view.shape = infer_shape(operation, parents, attrs);       
        
        auto stride_func = Registry<StrideInferFunc>::get(operation);
        if (stride_func) {
            this->view.strides = stride_func(parents, attrs, this->view.shape);
        } else {
            this->view.strides = calc_contiguous_strides(this->view.shape);
        }
    }

    bool Node::is_realized() const {
        return data != nullptr;
    }

    void Node::realize() {
        if (!is_realized()) {
            execute_graph(shared_from_this()); 
        }
    }
}