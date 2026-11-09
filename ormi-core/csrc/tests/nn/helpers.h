#pragma once
#include <algorithm>
#include <memory>
#include <utility>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/tensor.h>

namespace nn_test {
    inline std::shared_ptr<ormi::core::Tensor> leaf(ormi::core::Shape shape, double value = 1.0) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float64
        );
        std::fill_n(static_cast<double*>(tensor->data()), tensor->numel(), value);
        return tensor;
    }
}
