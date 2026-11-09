#pragma once
#include <cstddef>
#include <memory>

#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    struct MatmulShape {
        Shape batch;
        Shape output;
        std::size_t rows;
        std::size_t inner;
        std::size_t columns;
        bool lhs_vector;
        bool rhs_vector;
    };

    MatmulShape matmul_shape(const Shape& lhs, const Shape& rhs);

    // [..., M, K] @ [..., K, N] -> [..., M, N], with broadcast leading axes.
    // Rank-one operands are promoted to matrices for multiplication, then their
    // inserted axes are removed. Two vectors produce a scalar. Scalar inputs are rejected.
    std::shared_ptr<Tensor> matmul(
        const std::shared_ptr<Tensor>& lhs,
        const std::shared_ptr<Tensor>& rhs
    );
}
