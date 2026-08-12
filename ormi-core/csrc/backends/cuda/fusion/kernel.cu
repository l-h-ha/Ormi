#include <memory>
#include <span>
#include <utility>

#include <cuda_runtime_api.h>

#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/fusion/compile.h"
#include "backends/cuda/fusion/execute.h"
#include "backends/cuda/fusion/kernel.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda::fusion {
    bool supports(const core::runtime::fusion::Expr& expression) {
        if (!expression.auxiliary_outputs.empty()) {
            return !expression.reduction;
        }
        if (!expression.reduction) {
            return true;
        }

        const auto& reduction = *expression.reduction;
        return
            (reduction.op == core::ops::ReduceOp::sum ||
                reduction.op == core::ops::ReduceOp::mean) &&
            reduction.reduced.size != 0;
    }

    core::Setup prepare(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression
    ) {
        if (out.numel() == 0) {
            return {};
        }

        auto& cuda_state = state(out.device());
        check(cudaSetDevice(cuda_state.index()), "cudaSetDeviceFusion");
        auto compiled = compile(expression, cuda_state);
        return execution::prepare(
            out,
            auxiliary_outputs,
            expression,
            cuda_state,
            std::move(compiled)
        );
    }

    void execute(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression
    ) {
        if (out.numel() == 0) {
            return;
        }

        auto& cuda_state = state(out.device());
        check(cudaSetDevice(cuda_state.index()), "cudaSetDeviceFusion");
        auto compiled = compile(expression, cuda_state);
        execution::run(
            out,
            auxiliary_outputs,
            expression,
            cuda_state,
            std::move(compiled)
        );
    }
}
