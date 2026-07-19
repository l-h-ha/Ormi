#pragma once
#include <memory>

#include <ormi/core/attrs.h>
#include <ormi/core/device.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    std::shared_ptr<Tensor> full(Shape shape, Scalar value, Device device);

    std::shared_ptr<Tensor> zeros_like(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> ones_like(const std::shared_ptr<Tensor>& input);

    bool is_zero(const Tensor& tensor);
    bool is_one(const Tensor& tensor);
}
