#pragma once
#include <memory>

#include <ormi/core/tensor.h>

namespace ormi::nn::detail {
    // Both share value's storage, without allocating or copying an output.
    // depend adds execution order only. Its second parent receives no gradient.
    std::shared_ptr<core::Tensor> depend(
        const std::shared_ptr<core::Tensor>& value,
        const std::shared_ptr<core::Tensor>& dependency
    );
    // After all three parents run, copy replacement into a realized leaf destination.
    // Gradients pass only to value, retaining the update as an execution dependency.
    std::shared_ptr<core::Tensor> update(
        const std::shared_ptr<core::Tensor>& value,
        const std::shared_ptr<core::Tensor>& destination,
        const std::shared_ptr<core::Tensor>& replacement
    );
}
