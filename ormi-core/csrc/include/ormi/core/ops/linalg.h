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

    std::shared_ptr<Tensor> matmul(
        const std::shared_ptr<Tensor>& lhs,
        const std::shared_ptr<Tensor>& rhs
    );
}
