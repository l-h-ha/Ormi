#pragma once
#include <vector>
#include <memory>

#include "shape.h"
#include "ops.h"
#include "device.h"
#include "memory.h"
#include "attributes.h"

namespace ormi {
    struct Node : public std::enable_shared_from_this<Node> {
        ops::OpEnum op;
        std::vector<std::shared_ptr<Node>> parents;
        
        std::shared_ptr<Buffer> data;
        std::shared_ptr<Node> grad;
        
        bool requires_grad = false;
        Device device;
        TensorView view;
        OpAttributes attrs;
    
        // Leaf node constructor
        Node(std::shared_ptr<Buffer> mem, bool req_grad, Device dev);
    
        // Internal node constructor
        Node(
            ops::OpEnum operation, 
            std::vector<std::shared_ptr<Node>> parent_nodes, 
            Device dev,
            OpAttributes attributes = NoAttrs{}
        );
        
        bool is_realized() const;
        void realize();
    };
}