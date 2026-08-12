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
            Tensor& tensor,
            std::span<Tensor* const> auxiliary_outputs,
            const std::unordered_set<const Tensor*>* boundaries
        ) {
            const auto& selected = backend(tensor.device());
            if (!selected.fusion.execute) {
                return std::nullopt;
            }

            std::optional<fusion::Expr> expression;
            if (auxiliary_outputs.empty()) {
                expression = boundaries
                    ? fusion::build(tensor, *boundaries)
                    : fusion::build(tensor);
            } else {
                expression = boundaries
                    ? fusion::build(tensor, auxiliary_outputs, *boundaries)
                    : fusion::build(tensor, auxiliary_outputs);
            }

            if (
                !expression ||
                (
                    selected.fusion.supports &&
                    !selected.fusion.supports(*expression)
                )
            ) {
                return std::nullopt;
            }

            if (
                boundaries &&
                (selected.fusion.prepare || selected.fusion.prepare_many)
            ) {
                return Step(
                    tensor,
                    selected.fusion.prepare,
                    selected.fusion.prepare_many,
                    selected.fusion.execute,
                    std::move(*expression),
                    {
                        auxiliary_outputs.begin(),
                        auxiliary_outputs.end()
                    }
                );
            }
            return Step(
                tensor,
                nullptr,
                Setup{},
                selected.fusion.execute,
                std::move(expression),
                {
                    auxiliary_outputs.begin(),
                    auxiliary_outputs.end()
                }
            );
        }

        Step lower(
            Tensor& tensor,
            const std::unordered_set<const Tensor*>* boundaries
        ) {
            if (auto step = lower_fusion(tensor, {}, boundaries)) {
                return std::move(*step);
            }
            return Step(
                tensor,
                tensor.op()->compute,
                tensor.op()->prepare
                    ? tensor.op()->prepare(tensor)
                    : Setup{},
                nullptr,
                std::nullopt
            );
        }
    }

    Step lower(Tensor& tensor) {
        return lower(tensor, nullptr);
    }

    Step lower(
        Tensor& tensor,
        const std::unordered_set<const Tensor*>& boundaries
    ) {
        return lower(tensor, &boundaries);
    }

    std::optional<Step> lower_group(
        Tensor& tensor,
        std::span<Tensor* const> auxiliary_outputs
    ) {
        if (auxiliary_outputs.empty()) {
            return std::nullopt;
        }
        return lower_fusion(tensor, auxiliary_outputs, nullptr);
    }

    std::optional<Step> lower_group(
        Tensor& tensor,
        std::span<Tensor* const> auxiliary_outputs,
        const std::unordered_set<const Tensor*>& boundaries
    ) {
        if (auxiliary_outputs.empty()) {
            return std::nullopt;
        }
        return lower_fusion(tensor, auxiliary_outputs, &boundaries);
    }
}
