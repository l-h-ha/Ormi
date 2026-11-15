#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    // Select length elements along axis, starting at start with a nonzero step.
    // Positive steps share storage. Negative steps copy when the graph executes.
    std::shared_ptr<Tensor> slice(
        const std::shared_ptr<Tensor>& input,
        std::int64_t axis,
        std::int64_t start,
        std::int64_t length,
        std::int64_t step = 1
    );
    // Select one coordinate along axis and remove that dimension.
    std::shared_ptr<Tensor> select_axis(
        const std::shared_ptr<Tensor>& input, std::int64_t axis, std::int64_t index
    );
    // Join nonempty input lists along axis. Other dimensions, dtype, and device must match.
    std::shared_ptr<Tensor> concat(
        const std::vector<std::shared_ptr<Tensor>>& inputs, std::int64_t axis = 0
    );

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
