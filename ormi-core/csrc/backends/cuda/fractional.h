#pragma once

#include <ormi/core/attrs.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda {
    core::Setup prepare_fractional(
        core::ops::FractionalOp op,
        core::Tensor& out,
        const core::Caputo& attrs
    );

    void fractional(
        core::ops::FractionalOp op,
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor* initial_derivative,
        const core::Caputo& attrs
    );
}
