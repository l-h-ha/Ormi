#pragma once

#include <ormi/core/attrs.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cpu::caputo {
    void l1(
        core::ops::FractionalOp op,
        core::Tensor& out,
        const core::Tensor& input,
        const core::Caputo& attrs
    );
}
