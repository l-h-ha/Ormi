#include <cstdint>
#include <span>

#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/reduction.h"
#include "backends/cuda/reduction/sum.h"

namespace ormi::backends::cuda {
    void reduce(
        core::ops::ReduceOp op,
        core::Tensor& out,
        const core::Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    ) {
        // Core has already encoded keepdims in out.shape().
        static_cast<void>(keepdims);

        switch (op) {
            case core::ops::ReduceOp::sum:
                reduce_sum(out, input, axes);
                return;
            case core::ops::ReduceOp::mean:
                reduce_mean(out, input, axes);
                return;
        }

        ORMI_THROW(OperationError, "Unknown CUDA reduction operation.");
    }
}
