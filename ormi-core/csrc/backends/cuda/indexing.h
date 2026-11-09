#pragma once
#include <cstddef>

#include <ormi/core/attrs.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda {
    void scatter_scalar(core::Tensor& out, const core::Tensor& scalar, std::size_t index);

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
