#pragma once

#include <cstddef>

#include <ormi/core/tensor.h>

namespace ormi::backends::cpu {
    inline std::size_t storage_offset(std::size_t linear, const core::Tensor& tensor) {
        const auto& dims = tensor.shape().dims();
        const auto strides = tensor.strides();
        std::size_t offset = 0;

        for (std::size_t axis = tensor.rank(); axis > 0; --axis) {
            const auto index = axis - 1;
            const auto coordinate = linear % static_cast<std::size_t>(dims[index]);
            linear /= static_cast<std::size_t>(dims[index]);
            offset += coordinate * static_cast<std::size_t>(strides[index]);
        }

        return offset;
    }
}
