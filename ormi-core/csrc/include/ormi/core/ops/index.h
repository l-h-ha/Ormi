#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    // Scalar view at a row-major logical index. Uses the input strides and keeps
    // the parent alive without realizing or copying its data.
    std::shared_ptr<Tensor> select(const std::shared_ptr<Tensor>& input, std::size_t index);

    // Select one element along axis for every remaining coordinate. Int32 indices
    // have the input shape with axis removed. Invalid indices produce NaN.
    std::shared_ptr<Tensor> pick(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& indices,
        std::int64_t axis = -1,
        std::optional<std::int32_t> ignore_index = std::nullopt
    );

    // Transpose of pick: place each value at its index and zero the other entries.
    // Input and indices have shape with the selected axis removed.
    std::shared_ptr<Tensor> unpick(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& indices,
        Shape shape,
        std::int64_t axis = -1,
        std::optional<std::int32_t> ignore_index = std::nullopt
    );
}
