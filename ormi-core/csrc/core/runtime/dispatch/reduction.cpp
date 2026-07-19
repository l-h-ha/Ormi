#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/dispatch/reduction.h>

namespace ormi::core::runtime::reduction {
    void reduce(
        ops::ReduceOp op,
        Tensor& out,
        const Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    ) {
        out.device().backend()->reduce(op, out, input, axes, keepdims);
    }
}
