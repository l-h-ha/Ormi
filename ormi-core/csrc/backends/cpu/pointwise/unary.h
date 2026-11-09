#pragma once

#include <algorithm>
#include <cstddef>

#include <ormi/core/tensor.h>

#include "backends/cpu/index.h"

namespace ormi::backends::cpu {
    template <typename T, typename Apply>
    void unary(core::Tensor& out, const core::Tensor& input, Apply apply) {
        auto* out_data = static_cast<T*>(out.data());
        const auto* input_data = static_cast<const T*>(input.data());
        const auto elements = out.numel();

        if (input.is_contiguous()) {
            for (std::size_t i = 0; i < elements; ++i) {
                out_data[i] = apply(input_data[i]);
            }
            return;
        }

        if (elements == 0) {
            return;
        }

        const auto& dims = input.shape().dims();
        const auto strides = input.strides();
        std::size_t span = 1;
        std::size_t step = 0;
        auto outer_axes = input.rank();

        // Merge trailing axes while their elements keep a constant stride.
        while (outer_axes > 0) {
            const auto axis = outer_axes - 1;
            const auto extent = static_cast<std::size_t>(dims[axis]);
            if (extent == 1) {
                --outer_axes;
                continue;
            }
            if (span == 1) {
                step = static_cast<std::size_t>(strides[axis]);
            } else if (static_cast<std::size_t>(strides[axis]) != step * span) {
                break;
            }
            span *= extent;
            --outer_axes;
        }

        // Shared storage must keep sequential reads after each output write.
        const bool repeat_value = step == 0 && out.buffer() != input.buffer();
        for (std::size_t begin = 0; begin < elements; begin += span) {
            const auto offset = storage_offset(begin, input);
            if (step == 1) {
                for (std::size_t i = 0; i < span; ++i) {
                    out_data[begin + i] = apply(input_data[offset + i]);
                }
            } else if (repeat_value) {
                std::fill_n(out_data + begin, span, apply(input_data[offset]));
            } else {
                for (std::size_t i = 0; i < span; ++i) {
                    out_data[begin + i] = apply(input_data[offset + i * step]);
                }
            }
        }
    }
}
