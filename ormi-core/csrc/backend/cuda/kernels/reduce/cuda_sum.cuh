#pragma once
#include <memory>

namespace ormi::cuda {
    std::shared_ptr<Buffer> sum(
        std::shared_ptr<Buffer> in_buf, 
        const TensorView& in_view,
        const std::vector<int>& axes, 
        const TensorView& out_view,
        const bool keepdims
    );
    
    void sum_handler(std::shared_ptr<Node> root);
}