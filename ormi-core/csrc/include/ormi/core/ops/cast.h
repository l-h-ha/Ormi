#pragma once

#include <memory>

#include <ormi/core/dtype.h>

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::ops {
    std::shared_ptr<Tensor> cast(const std::shared_ptr<Tensor>& input, DType dtype);
}
