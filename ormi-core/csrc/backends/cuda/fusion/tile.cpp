#include <algorithm>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "backends/cuda/fusion/tile.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        constexpr std::size_t warp_width = 32;
        constexpr std::size_t minimum_outputs = warp_width * 8;

        std::optional<std::size_t> dimension(
            std::size_t logical_stride,
            std::size_t extent,
            const std::vector<std::size_t>& shape
        ) {
            std::size_t stride = 1;
            for (std::size_t i = shape.size(); i > 0; --i) {
                const auto current = i - 1;
                if (
                    stride == logical_stride &&
                    shape[current] == extent
                ) {
                    return current;
                }
                stride *= shape[current];
            }
            return std::nullopt;
        }

        std::optional<std::vector<std::size_t>> dimensions(
            const core::runtime::fusion::Iteration& iteration,
            const std::vector<std::size_t>& shape
        ) {
            std::vector<std::size_t> result;
            result.reserve(iteration.axes.size());
            for (const auto& axis : iteration.axes) {
                const auto current =
                    dimension(axis.stride, axis.extent, shape);
                if (!current) {
                    return std::nullopt;
                }
                result.push_back(*current);
            }
            return result;
        }

        std::vector<std::size_t> contiguous_strides(
            const std::vector<std::size_t>& shape
        ) {
            std::vector<std::size_t> result(shape.size());
            std::size_t stride = 1;
            for (std::size_t i = shape.size(); i > 0; --i) {
                result[i - 1] = stride;
                stride *= shape[i - 1];
            }
            return result;
        }
    }

    std::optional<Tile> tile(
        const core::runtime::fusion::Expr& expression
    ) {
        if (
            !expression.reduction ||
            expression.reduction->output.size < minimum_outputs
        ) {
            return std::nullopt;
        }

        const auto output =
            dimensions(expression.reduction->output, expression.shape);
        const auto reduced =
            dimensions(expression.reduction->reduced, expression.shape);
        if (
            !output ||
            !reduced ||
            reduced->empty() ||
            output->size() + reduced->size() != expression.shape.size()
        ) {
            return std::nullopt;
        }

        std::vector<bool> represented(expression.shape.size(), false);
        for (const auto current : *output) {
            if (represented[current]) {
                return std::nullopt;
            }
            represented[current] = true;
        }
        for (const auto current : *reduced) {
            if (represented[current]) {
                return std::nullopt;
            }
            represented[current] = true;
        }
        if (
            !std::all_of(
                represented.begin(),
                represented.end(),
                [](bool value) {
                    return value;
                }
            )
        ) {
            return std::nullopt;
        }

        std::vector<std::vector<std::size_t>> inputs(
            expression.inputs.size(),
            std::vector<std::size_t>(expression.shape.size(), 0)
        );
        for (const auto& node : expression.nodes) {
            const auto* input =
                std::get_if<core::runtime::fusion::Input>(
                    &node.instruction
                );
            if (!input) {
                continue;
            }
            if (input->index >= inputs.size()) {
                return std::nullopt;
            }
            if (input->linear) {
                inputs[input->index] =
                    contiguous_strides(expression.shape);
                continue;
            }
            for (const auto& axis : input->axes) {
                const auto current = dimension(
                    axis.divisor,
                    axis.extent,
                    expression.shape
                );
                if (!current) {
                    return std::nullopt;
                }
                inputs[input->index][*current] = axis.stride;
            }
        }

        std::optional<std::size_t> tiled_axis;
        for (const auto current : *output) {
            if (expression.shape[current] < warp_width) {
                continue;
            }
            const bool contiguous = std::all_of(
                inputs.begin(),
                inputs.end(),
                [current](const std::vector<std::size_t>& strides) {
                    return strides[current] <= 1;
                }
            );
            const bool reads_axis = std::any_of(
                inputs.begin(),
                inputs.end(),
                [current](const std::vector<std::size_t>& strides) {
                    return strides[current] == 1;
                }
            );
            if (
                contiguous &&
                reads_axis &&
                (
                    !tiled_axis ||
                    expression.shape[current] >
                        expression.shape[*tiled_axis]
                )
            ) {
                tiled_axis = current;
            }
        }
        if (!tiled_axis) {
            return std::nullopt;
        }

        const auto extent = expression.shape[*tiled_axis];
        const auto tiles =
            (extent + warp_width - 1) / warp_width;
        const auto outer =
            expression.reduction->output.size / extent;
        return Tile{
            .axis = *tiled_axis,
            .width = warp_width,
            .tasks = tiles * outer,
            .output = *output,
            .reduced = *reduced,
            .inputs = std::move(inputs)
        };
    }
}
