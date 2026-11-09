#pragma once
#include <memory>

#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::nn {
    // Normalize the trailing normalized_shape axes using population variance.
    // Float16 uses Float32 statistics. The output keeps the input dtype.
    // Optional weight and bias match the normalized shape, input dtype, and input device.
    // Float16 inputs also accept Float32 weight and bias.
    std::shared_ptr<core::Tensor> layer_norm(
        const std::shared_ptr<core::Tensor>& input,
        const core::Shape& normalized_shape,
        const std::shared_ptr<core::Tensor>& weight = nullptr,
        const std::shared_ptr<core::Tensor>& bias = nullptr,
        double eps = 1e-5
    );
}
