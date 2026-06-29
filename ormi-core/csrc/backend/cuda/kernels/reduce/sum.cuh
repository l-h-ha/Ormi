#pragma once
#include <memory>
#include "core/buffer.h"
#include "core/shape.h"

namespace ormi::cuda {
    std::shared_ptr<Buffer> sum(
        const std::shared_ptr<Buffer>& in_buf, 
        const TensorView& in_view,
        const std::vector<int>& axes, 
        const TensorView& out_view,
        const bool keepdims,
        std::shared_ptr<Buffer> out_buffer = nullptr
    );
    
    void sum_handler(const std::shared_ptr<Node>& root);
}