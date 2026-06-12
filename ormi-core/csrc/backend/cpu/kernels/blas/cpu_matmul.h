#pragma once
#include <memory>

namespace ormi::cpu {
    std::shared_ptr<Buffer> matmul(
        std::shared_ptr<Buffer> A, 
        const TensorView& view_A,
        std::shared_ptr<Buffer> B, 
        const TensorView& view_B,
        const TensorView& out_view
    );

    void matmul_handler(std::shared_ptr<Node> root);
}