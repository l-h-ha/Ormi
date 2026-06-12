#pragma once
#include <memory>
#include "core/node.h"

namespace ormi::cuda {
    std::shared_ptr<Buffer> matmul(
        std::shared_ptr<Buffer> A, 
        const TensorView& view_A, 
        std::shared_ptr<Buffer> B, 
        const TensorView& view_B, 
        const TensorView& out_view
    );

    void matmul_handler(std::shared_ptr<Node> root);
}
