#pragma once
#include <memory>

namespace ormi::cpu {
    std::shared_ptr<Buffer> matmul(
        const std::shared_ptr<Buffer>& A, 
        const TensorView& view_A,
        const std::shared_ptr<Buffer>& B, 
        const TensorView& view_B,
        const TensorView& out_view,
        std::shared_ptr<Buffer> out_buffer = nullptr
    );

    void matmul_handler(const std::shared_ptr<Node>& root);
}