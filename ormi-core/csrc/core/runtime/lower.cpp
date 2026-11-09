#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ormi/backend/api.h>
#include <ormi/core/op.h>
#include <ormi/core/tensor.h>

#include "core/runtime/device.h"
#include "core/runtime/fusion/expr.h"
#include "core/runtime/lower.h"

namespace ormi::core::runtime {
    namespace {
        std::optional<Step> lower_fusion(
            Tensor& output,
            std::span<Tensor* const> auxiliary_outputs,
            const std::unordered_set<const Tensor*>* boundaries,
            Preparation preparation
        ) {
            const auto& target = backend(output.device()).fusion;
            if (!target.execute) {
                return std::nullopt;
            }

            std::optional<fusion::Expr> expression;
            if (auxiliary_outputs.empty()) {
                expression = boundaries
                    ? fusion::build(output, *boundaries)
                    : fusion::build(output);
            } else {
                expression = boundaries
                    ? fusion::build(output, auxiliary_outputs, *boundaries)
                    : fusion::build(output, auxiliary_outputs);
            }

            if (
                !expression ||
                (target.supports && !target.supports(*expression))
            ) {
                return std::nullopt;
            }

            std::vector<Tensor*> outputs{
                auxiliary_outputs.begin(),
                auxiliary_outputs.end()
            };
            if (preparation == Preparation::deferred && (target.prepare || target.prepare_many)) {
                return Step::deferred_fusion(
                    output,
                    target.prepare,
                    target.prepare_many,
                    target.execute,
                    std::move(*expression),
                    std::move(outputs)
                );
            }
            return Step::fusion(
                output,
                target.execute,
                std::move(*expression),
                std::move(outputs)
            );
        }

        Step lower(
            Tensor& tensor,
            const fusion::Candidates* candidates,
            const std::unordered_set<const Tensor*>* boundaries,
            Preparation preparation
        ) {
            if (candidates) {
                const auto group = candidates->find(&tensor);
                if (group != candidates->end() && !group->second.empty()) {
                    if (auto step = lower_fusion(tensor, group->second, boundaries, preparation)) {
                        return std::move(*step);
                    }
                }
            }
            if (auto step = lower_fusion(tensor, {}, boundaries, preparation)) {
                return std::move(*step);
            }
            if (tensor.op()->output == Output::alias) {
                return Step::alias(tensor);
            }
            return Step::compute(
                tensor,
                tensor.op()->compute,
                tensor.op()->prepare
                    ? tensor.op()->prepare(tensor)
                    : Setup{}
            );
        }
    }

    Step lower(Tensor& tensor, Preparation preparation) {
        return lower(tensor, nullptr, nullptr, preparation);
    }

    Step lower(Tensor& tensor, const fusion::Candidates& candidates, Preparation preparation) {
        return lower(tensor, &candidates, nullptr, preparation);
    }

    Step lower(
        Tensor& tensor,
        const fusion::Candidates& candidates,
        const std::unordered_set<const Tensor*>& boundaries,
        Preparation preparation
    ) {
        return lower(tensor, &candidates, &boundaries, preparation);
    }
}
