#include <algorithm>
#include <cstddef>
#include <utility>
#include <variant>
#include <vector>

#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/order.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        bool dense(const core::Tensor& tensor, std::size_t elements) {
            if (tensor.numel() != elements) {
                return false;
            }

            std::vector<std::pair<std::size_t, std::size_t>> dimensions;
            dimensions.reserve(tensor.rank());
            for (std::size_t i = 0; i < tensor.rank(); ++i) {
                const auto extent =
                    static_cast<std::size_t>(tensor.shape()[i]);
                if (extent <= 1) {
                    continue;
                }

                const auto stride = tensor.strides()[i];
                if (stride <= 0) {
                    return false;
                }
                dimensions.emplace_back(
                    static_cast<std::size_t>(stride),
                    extent
                );
            }
            std::sort(dimensions.begin(), dimensions.end());

            std::size_t expected = 1;
            for (const auto& [stride, extent] : dimensions) {
                if (stride != expected) {
                    return false;
                }
                expected *= extent;
            }
            return expected == elements;
        }
    }

    bool physical_order(
        const core::runtime::fusion::Expr& expression
    ) {
        if (
            !expression.reduction ||
            expression.reduction->output.size != 1 ||
            expression.reduction->reduced.size != expression.elements ||
            expression.inputs.empty()
        ) {
            return false;
        }

        const core::runtime::fusion::Input* first = nullptr;
        std::vector<bool> found(expression.inputs.size(), false);
        for (const auto& node : expression.nodes) {
            const auto* input =
                std::get_if<core::runtime::fusion::Input>(
                    &node.instruction
                );
            if (!input) {
                continue;
            }
            if (
                input->index >= expression.inputs.size() ||
                !dense(*expression.inputs[input->index], expression.elements)
            ) {
                return false;
            }
            if (
                first &&
                (
                    input->linear != first->linear ||
                    input->axes != first->axes
                )
            ) {
                return false;
            }
            first = first ? first : input;
            found[input->index] = true;
        }

        return
            first &&
            std::all_of(
                found.begin(),
                found.end(),
                [](bool value) {
                    return value;
                }
            );
    }
}
