#pragma once

#include <cstddef>

#include <ormi/core/tensor.h>

namespace ormi::backends::cpu {
    inline std::size_t storage_offset(std::size_t linear, const core::Tensor& tensor) {
        const auto& dims = tensor.shape().dims();
        const auto strides = tensor.strides();
        std::size_t offset = 0;

        for (std::size_t axis = tensor.rank(); axis > 0; --axis) {
            const auto input_axis = axis - 1;
            const auto extent = static_cast<std::size_t>(dims[input_axis]);
            const auto coordinate = linear % extent;
            linear /= extent;
            offset += coordinate * static_cast<std::size_t>(strides[input_axis]);
        }

        return offset;
    }
}
