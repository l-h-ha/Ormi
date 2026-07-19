#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& input, Shape shape);
    std::shared_ptr<Tensor> transpose(
        const std::shared_ptr<Tensor>& input,
        std::vector<std::int64_t> axes
    );
    std::shared_ptr<Tensor> expand(const std::shared_ptr<Tensor>& input, Shape shape);
}
