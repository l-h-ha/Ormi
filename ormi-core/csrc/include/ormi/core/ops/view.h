#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    // Views share the input's storage. These calls build metadata nodes without moving data.
    std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& input, Shape shape);
    // axes lists every input axis exactly once, in the desired output order.
    std::shared_ptr<Tensor> transpose(
        const std::shared_ptr<Tensor>& input,
        std::vector<std::int64_t> axes
    );
    // Broadcast dimensions use zero strides, so several logical elements can share one value.
    std::shared_ptr<Tensor> expand(const std::shared_ptr<Tensor>& input, Shape shape);
}
