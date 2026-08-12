#pragma once
#include <span>

#include <ormi/backend/api.h>
#include <ormi/backend/fusion.h>
#include <ormi/core/op.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cpu::jit {
    bool supports(const core::runtime::fusion::Expr& expression);
    core::Setup prepare(
        core::Tensor& out,
        const core::runtime::fusion::Expr& expression
    );
    core::Setup prepare(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression
    );
    void prepare_many(std::span<core::runtime::FusionJob> jobs);
    void execute(core::Tensor& out, const core::runtime::fusion::Expr& expression);
    void execute(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression
    );
}
