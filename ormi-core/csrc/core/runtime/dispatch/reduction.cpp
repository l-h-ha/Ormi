#include <ormi/backend/api.h>

#include "core/runtime/dispatch/reduction.h"
#include "core/runtime/device.h"

namespace ormi::core::runtime::reduction {
    void reduce(
        ops::ReduceOp op,
        Tensor& out,
        const Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    ) {
        backend(out.device()).ops.reduce(
            op,
            out,
            input,
            axes,
            keepdims
        );
    }
}
