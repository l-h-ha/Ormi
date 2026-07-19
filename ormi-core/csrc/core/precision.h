#pragma once

#include <ormi/core/dtype.h>
#include <ormi/core/op.h>

namespace ormi::core::precision {
    struct State {
        bool enabled = false;
        DType dtype = DType::float16;
    };

    extern thread_local State state;

    inline DType apply(Precision precision, DType dtype) noexcept {
        if (!state.enabled) {
            return dtype;
        }

        switch (precision) {
            case Precision::preserve:
                return dtype;
            case Precision::lower:
                return dtype == DType::float32 ? state.dtype : dtype;
            case Precision::float32:
                return dtype == DType::float16 ? DType::float32 : dtype;
        }

        return dtype;
    }
}
