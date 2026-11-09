#pragma once
#include <ormi/core/attrs.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda {
    void bernoulli(core::Tensor& out, const core::Bernoulli& attrs);
}
