#pragma once

#include <ormi/core/attrs.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cpu {
    void pick(
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor& indices,
        const core::Pick& attrs
    );
    void unpick(
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor& indices,
        const core::Pick& attrs
    );
}
