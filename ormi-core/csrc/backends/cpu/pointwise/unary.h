#pragma once

#include <ormi/core/tensor.h>

#include "backends/cpu/index.h"

namespace ormi::backends::cpu {
    template <typename T, typename Apply>
    void unary(core::Tensor& out, const core::Tensor& t, Apply apply) {
        auto* out_data = static_cast<T*>(out.data());
        const auto* t_data = static_cast<const T*>(t.data());

        if (t.is_contiguous()) {
            for (std::size_t i = 0; i < out.numel(); ++i) {
                out_data[i] = apply(t_data[i]);
            }
            return;
        }

        for (std::size_t i = 0; i < out.numel(); ++i) {
            out_data[i] = apply(t_data[storage_offset(i, t)]);
        }
    }
}
