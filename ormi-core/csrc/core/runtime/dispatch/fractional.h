#pragma once

#include <ormi/core/attrs.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

namespace ormi::core::runtime::fractional {
    Setup prepare(
        ops::FractionalOp op,
        Tensor& out,
        const Caputo& attrs
    );

    void apply(
        ops::FractionalOp op,
        Tensor& out,
        const Tensor& input,
        const Tensor* initial_derivative,
        const Caputo& attrs
    );
}
