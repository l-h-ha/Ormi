#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <ormi/backend/api.h>
#include <ormi/core/setup.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/execute.h"
#include "backends/cpu/jit/kernel.h"

namespace ormi::backends::cpu::jit {
    bool supports(const core::runtime::fusion::Expr& expression) {
        if (!expression.auxiliary_outputs.empty()) {
            return !expression.reduction;
        }
        if (!expression.reduction) {
            return true;
        }

        return
            expression.reduction->op == core::ops::ReduceOp::sum ||
            expression.reduction->op == core::ops::ReduceOp::mean;
    }

    core::Setup prepare(
        core::Tensor& out,
        const core::runtime::fusion::Expr& expression
    ) {
        return prepare(out, {}, expression);
    }

    core::Setup prepare(
        core::Tensor&,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression
    ) {
        auto compiled = compile(expression);
        return execution::prepare(
            auxiliary_outputs,
            expression,
            std::move(compiled)
        );
    }

    void prepare_many(std::span<core::runtime::FusionJob> jobs) {
        std::vector<const core::runtime::fusion::Expr*> expressions;
        expressions.reserve(jobs.size());
        for (const auto& job : jobs) {
            expressions.push_back(job.expression);
        }

        auto compiled = compile(expressions);
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            *jobs[i].setup = execution::prepare(
                {
                    jobs[i].auxiliary_outputs,
                    jobs[i].auxiliary_count
                },
                *jobs[i].expression,
                std::move(compiled[i])
            );
        }
    }

    void execute(
        core::Tensor& out,
        const core::runtime::fusion::Expr& expression
    ) {
        execute(out, {}, expression);
    }

    void execute(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression
    ) {
        const auto compiled = compile(expression);
        execution::run(out, auxiliary_outputs, expression, *compiled);
    }
}
