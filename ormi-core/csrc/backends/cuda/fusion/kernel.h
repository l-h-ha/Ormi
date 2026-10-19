#pragma once
#include <span>

#include <ormi/backend/fusion.h>
#include <ormi/core/setup.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda::fusion {
    bool supports(const core::runtime::fusion::Expr& expression);
    core::Setup prepare(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression
    );
    void execute(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression
    );
}
