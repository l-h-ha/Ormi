#include <cstdint>
#include <span>

#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/reduction.h"
#include "backends/cpu/reduction/sum.h"

namespace ormi::backends::cpu {
    void reduce(
        core::ops::ReduceOp op,
        core::Tensor& out,
        const core::Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    ) {
        switch (op) {
            case core::ops::ReduceOp::sum:
                reduce_sum(out, input, axes, keepdims);
                return;
        }

        ORMI_THROW(OperationError, "Unknown CPU reduction operation.");
    }
}
