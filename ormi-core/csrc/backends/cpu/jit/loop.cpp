#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/loop.h"
#include "backends/cpu/jit/math.h"

namespace ormi::backends::cpu::jit {
    namespace {
        std::size_t lanes(core::DType dtype) {
            return dtype == core::DType::float64 ? 4 : 8;
        }

        bool contiguous(const core::runtime::fusion::Iteration& iteration) {
            return std::all_of(
                iteration.axes.begin(),
                iteration.axes.end(),
                [](const core::runtime::fusion::Axis& axis) {
                    return axis.stride == axis.divisor;
                }
            );
        }

        bool broadcast_view(const core::Tensor& tensor) {
            if (tensor.is_contiguous()) {
                return false;
            }

            const auto strides = tensor.strides();
            std::size_t expected = 1;
            bool broadcast = false;

            for (std::size_t i = tensor.rank(); i > 0; --i) {
                const auto axis = i - 1;
                if (strides[axis] == 0) {
                    broadcast = true;
                    continue;
                }
                if (static_cast<std::size_t>(strides[axis]) != expected) {
                    return false;
                }
                expected *= static_cast<std::size_t>(tensor.shape()[axis]);
            }
            return broadcast;
        }

        bool uses_float16(const core::runtime::fusion::Expr& expression) {
            return std::any_of(
                expression.nodes.begin(),
                expression.nodes.end(),
                [](const core::runtime::fusion::Node& node) {
                    return node.dtype == core::DType::float16;
                }
            );
        }

        bool vectorizable_arithmetic(
            const core::runtime::fusion::Expr& expression
        ) {
            if (
                expression.dtype != core::DType::float32 &&
                expression.dtype != core::DType::float64
            ) {
                return false;
            }

            return std::none_of(
                expression.nodes.begin(),
                expression.nodes.end(),
                [](const core::runtime::fusion::Node& node) {
                    const auto* unary =
                        std::get_if<core::runtime::fusion::Unary>(
                            &node.instruction
                        );
                    return unary && unary->op != core::ops::UnaryOp::neg;
                }
            );
        }

        bool vectorizable_math(
            const core::runtime::fusion::Expr& expression
        ) {
            if (
                expression.dtype != core::DType::float32 ||
                expression.reduction
            ) {
                return false;
            }

            bool found = false;
            for (const auto& node : expression.nodes) {
                if (node.dtype != core::DType::float32) {
                    return false;
                }
                const auto* unary =
                    std::get_if<core::runtime::fusion::Unary>(
                        &node.instruction
                    );
                if (unary && unary->op != core::ops::UnaryOp::neg) {
                    found = true;
                }
            }
            return found;
        }

        std::optional<std::size_t> dimension(
            std::size_t logical_stride,
            std::size_t extent,
            const std::vector<std::size_t>& shape
        ) {
            std::size_t divisor = 1;
            for (std::size_t i = shape.size(); i > 0; --i) {
                const auto dimension = i - 1;
                if (
                    divisor == logical_stride &&
                    shape[dimension] == extent
                ) {
                    return dimension;
                }
                divisor *= shape[dimension];
            }
            return std::nullopt;
        }

        std::optional<std::vector<std::vector<std::size_t>>> strided_strides(
            const core::runtime::fusion::Expr& expression
        ) {
            if (expression.shape.empty()) {
                return std::nullopt;
            }

            std::vector<std::vector<std::size_t>> result(
                expression.inputs.size(),
                std::vector<std::size_t>(expression.shape.size(), 0)
            );
            for (const auto& node : expression.nodes) {
                const auto* input =
                    std::get_if<core::runtime::fusion::Input>(&node.instruction);
                if (!input) {
                    continue;
                }
                if (input->index >= result.size()) {
                    return std::nullopt;
                }
                if (input->linear) {
                    std::size_t stride = 1;
                    for (std::size_t i = expression.shape.size(); i > 0; --i) {
                        result[input->index][i - 1] = stride;
                        stride *= expression.shape[i - 1];
                    }
                    continue;
                }
                for (const auto& axis : input->axes) {
                    const auto output_dimension =
                        dimension(
                            axis.divisor,
                            axis.extent,
                            expression.shape
                        );
                    if (!output_dimension) {
                        return std::nullopt;
                    }
                    result[input->index][*output_dimension] = axis.stride;
                }
            }
            return result;
        }

        std::optional<std::vector<std::size_t>> dimensions(
            const core::runtime::fusion::Iteration& iteration,
            const std::vector<std::size_t>& shape
        ) {
            std::vector<std::size_t> result;
            result.reserve(iteration.axes.size());
            for (const auto& axis : iteration.axes) {
                const auto value = dimension(
                    axis.stride,
                    axis.extent,
                    shape
                );
                if (!value) {
                    return std::nullopt;
                }
                result.push_back(*value);
            }
            return result;
        }

        std::size_t stride_cost(
            const std::vector<std::vector<std::size_t>>& strides,
            std::size_t dimension
        ) {
            std::size_t result = 0;
            for (const auto& input : strides) {
                result = std::max(result, input[dimension]);
            }
            return result;
        }

        bool plan_direct_reduction(
            const core::runtime::fusion::Expr& expression,
            Loop& loop
        ) {
            if (!expression.reduction) {
                return false;
            }

            const auto strides = strided_strides(expression);
            const auto output = dimensions(
                expression.reduction->output,
                expression.shape
            );
            const auto reduced = dimensions(
                expression.reduction->reduced,
                expression.shape
            );
            const bool empty_shape = std::any_of(
                expression.shape.begin(),
                expression.shape.end(),
                [](std::size_t extent) {
                    return extent == 0;
                }
            );
            if (
                !strides ||
                !output ||
                !reduced ||
                empty_shape ||
                output->size() + reduced->size() !=
                    expression.shape.size()
            ) {
                return false;
            }

            loop.strided = true;
            loop.strides = *strides;
            loop.output_dimensions = *output;
            loop.reduced_dimensions = *reduced;

            if (
                expression.reduction->output.size == 1 &&
                !loop.reduced_dimensions.empty()
            ) {
                const auto largest = std::max_element(
                    loop.reduced_dimensions.begin(),
                    loop.reduced_dimensions.end(),
                    [&expression, &loop](
                        std::size_t lhs,
                        std::size_t rhs
                    ) {
                        if (expression.shape[lhs] != expression.shape[rhs]) {
                            return expression.shape[lhs] <
                                expression.shape[rhs];
                        }
                        return stride_cost(loop.strides, lhs) <
                            stride_cost(loop.strides, rhs);
                    }
                );
                loop.split = *largest;
                std::stable_sort(
                    loop.reduced_dimensions.begin(),
                    loop.reduced_dimensions.end(),
                    [&loop](std::size_t lhs, std::size_t rhs) {
                        return stride_cost(loop.strides, lhs) >
                            stride_cost(loop.strides, rhs);
                    }
                );
                loop.partial = true;
                loop.unit = 1;

                const auto inner = loop.reduced_dimensions.back();
                const auto width = lanes(expression.dtype);
                const bool unit_stride = std::all_of(
                    loop.strides.begin(),
                    loop.strides.end(),
                    [inner](const std::vector<std::size_t>& input) {
                        return input[inner] <= 1;
                    }
                );
                if (
                    vectorizable_arithmetic(expression) &&
                    unit_stride &&
                    expression.shape[inner] >= width * 8
                ) {
                    loop.vectorized = true;
                    loop.lanes = width;
                    if (loop.split == inner) {
                        loop.unit = width;
                    }
                }

                bool unit_axis = false;
                for (const auto dimension : loop.reduced_dimensions) {
                    unit_axis = std::all_of(
                        loop.strides.begin(),
                        loop.strides.end(),
                        [dimension](
                            const std::vector<std::size_t>& input
                        ) {
                            return input[dimension] <= 1;
                        }
                    );
                    if (unit_axis) {
                        break;
                    }
                }
                constexpr std::size_t tile = 16;
                std::optional<std::size_t> tile_axis;
                if (!unit_axis && vectorizable_arithmetic(expression)) {
                    for (const auto dimension : loop.reduced_dimensions) {
                        if (
                            dimension == loop.split ||
                            expression.shape[dimension] < tile * 2
                        ) {
                            continue;
                        }
                        const bool serves_input = std::any_of(
                            loop.strides.begin(),
                            loop.strides.end(),
                            [dimension](
                                const std::vector<std::size_t>& input
                            ) {
                                return input[dimension] <= 1;
                            }
                        );
                        if (
                            serves_input &&
                            (
                                !tile_axis ||
                                expression.shape[dimension] >
                                    expression.shape[*tile_axis]
                            )
                        ) {
                            tile_axis = dimension;
                        }
                    }
                }
                if (
                    tile_axis &&
                    expression.shape[*loop.split] >= tile * 2
                ) {
                    loop.tile = tile;
                    loop.unit = loop.tile;
                    loop.tile_axis = tile_axis;
                }
            }

            bool tile_output_axis = false;
            if (
                !loop.output_dimensions.empty() &&
                !loop.reduced_dimensions.empty()
            ) {
                const auto output_axis = loop.output_dimensions[0];
                const auto tile =
                    expression.shape.size() == 2 ? 32u : 16u;
                tile_output_axis =
                    expression.shape[output_axis] >= tile &&
                    std::all_of(
                        loop.strides.begin(),
                        loop.strides.end(),
                        [output_axis](const std::vector<std::size_t>& input) {
                            return input[output_axis] <= 1;
                        }
                    );
                if (tile_output_axis) {
                    loop.tile = tile;
                }
            }
            if (tile_output_axis) {
                loop.vectorized = true;
                loop.tile_axis = loop.output_dimensions[0];
            } else if (
                expression.reduction->output.size > 1 &&
                !loop.reduced_dimensions.empty()
            ) {
                const auto inner = loop.reduced_dimensions.back();
                const auto width = lanes(expression.dtype);
                const bool unit_stride = std::all_of(
                    loop.strides.begin(),
                    loop.strides.end(),
                    [inner](const std::vector<std::size_t>& input) {
                        return input[inner] <= 1;
                    }
                );
                if (
                    vectorizable_arithmetic(expression) &&
                    unit_stride &&
                    expression.shape[inner] >= width * 8
                ) {
                    loop.vectorized = true;
                    loop.lanes = width;
                }
            }
            if (
                expression.reduction->output.size > 1 &&
                !loop.output_dimensions.empty()
            ) {
                const auto slab =
                    expression.reduction->output.size /
                    expression.shape[loop.output_dimensions.front()];
                const auto unit =
                    loop.tile != 0 ? loop.tile * slab : slab;
                if (expression.reduction->output.size > unit) {
                    loop.ranged = true;
                    loop.unit = unit;
                }
            }
            return true;
        }

        bool benefits_from_2d_tiling(
            const core::runtime::fusion::Expr& expression,
            const std::vector<std::vector<std::size_t>>& strides
        ) {
            if (
                expression.shape.size() != 2 ||
                expression.shape[0] < 32 ||
                expression.shape[1] < 32
            ) {
                return false;
            }

            return std::any_of(
                strides.begin(),
                strides.end(),
                [](const std::vector<std::size_t>& input) {
                    return input.size() == 2 &&
                        input[0] == 1 &&
                        input[1] > 1;
                }
            );
        }
    }

    std::size_t parallel_work(
        const core::runtime::fusion::Expr& expression,
        bool paired_math
    ) noexcept {
        constexpr std::size_t transcendental_weight = 16;
        constexpr auto limit = std::numeric_limits<std::size_t>::max();
        std::size_t weight = 1;

        for (std::size_t i = 0; i < expression.nodes.size(); ++i) {
            const auto& node = expression.nodes[i];
            const auto* unary = std::get_if<core::runtime::fusion::Unary>(
                &node.instruction
            );
            if (!unary || unary->op == core::ops::UnaryOp::neg) {
                continue;
            }

            if (paired_math) {
                const auto paired = trig_pair(expression, i);
                if (paired && *paired < i) {
                    continue;
                }
            }

            constexpr auto extra = transcendental_weight - 1;
            if (weight > limit - extra) {
                return limit;
            }
            weight += extra;
        }

        if (expression.elements > limit / weight) {
            return limit;
        }
        return expression.elements * weight;
    }

    Loop plan(const core::runtime::fusion::Expr& expression) {
        Loop result;
        result.inputs.assign(expression.inputs.size(), Access::indexed);

        const bool packed_float16 = uses_float16(expression);
        const bool packed_math = vectorizable_math(expression);
        const bool has_noncontiguous_input = std::any_of(
            expression.inputs.begin(),
            expression.inputs.end(),
            [](const core::Tensor* input) {
                return !input->is_contiguous();
            }
        );
        const bool has_strided_input = has_noncontiguous_input && std::any_of(
            expression.inputs.begin(),
            expression.inputs.end(),
            [](const core::Tensor* input) {
                return !input->is_contiguous() && !broadcast_view(*input);
            }
        );

        if (
            expression.reduction &&
            (
                has_noncontiguous_input ||
                !contiguous(expression.reduction->reduced)
            )
        ) {
            if (plan_direct_reduction(expression, result)) {
                return result;
            }
            if (has_strided_input) {
                return result;
            }
        }

        if (has_strided_input) {
            if (auto strides = strided_strides(expression)) {
                result.strided = true;
                if (benefits_from_2d_tiling(expression, *strides)) {
                    result.tile = 32;
                }
                result.strides = std::move(*strides);
            }
            return result;
        }

        if (expression.reduction && expression.reduction->output.size == 1) {
            result.partial = true;
            result.unit = 1;
        }

        if (expression.reduction && expression.reduction->output.size > 1) {
            result.ranged = true;
            result.unit = 1;
        }

        std::optional<std::size_t> span;
        bool broadcast = false;
        bool vectorizable = true;

        for (std::size_t index = 0; index < expression.nodes.size(); ++index) {
            const auto& node = expression.nodes[index];
            if (packed_float16) {
                if (
                    node.dtype != core::DType::float16 &&
                    node.dtype != core::DType::float32
                ) {
                    return result;
                }
            } else if (expression.reduction && node.dtype != expression.dtype) {
                return result;
            }

            if (const auto* cast =
                    std::get_if<core::runtime::fusion::Cast>(&node.instruction)) {
                if (!packed_float16) {
                    continue;
                }
                if (cast->input >= index) {
                    return result;
                }
                const auto source = expression.nodes[cast->input].dtype;
                if (
                    source != core::DType::float16 &&
                    source != core::DType::float32
                ) {
                    return result;
                }
            }

            if (const auto* unary =
                    std::get_if<core::runtime::fusion::Unary>(&node.instruction);
                unary && unary->op != core::ops::UnaryOp::neg) {
                vectorizable = false;
            }

            const auto* input =
                std::get_if<core::runtime::fusion::Input>(&node.instruction);
            if (!input) {
                continue;
            }
            if (input->index >= result.inputs.size()) {
                ORMI_THROW(
                    OperationError,
                    "Fusion expression references an invalid input."
                );
            }

            if (input->linear) {
                result.inputs[input->index] = Access::linear;
                continue;
            }
            if (input->axes.empty()) {
                result.inputs[input->index] = Access::scalar;
                continue;
            }

            broadcast = true;
            for (const auto& axis : input->axes) {
                if (axis.divisor == 1 && axis.stride != 1) {
                    return result;
                }

                const auto limit = axis.divisor == 1
                    ? axis.extent
                    : axis.divisor;
                if (limit == 0) {
                    return result;
                }
                span = span ? std::min(*span, limit) : limit;
            }
        }

        if (packed_float16 && expression.reduction) {
            if (!vectorizable || broadcast) {
                return result;
            }
            if (
                !contiguous(expression.reduction->reduced) ||
                expression.reduction->reduced.size < 64
            ) {
                return result;
            }

            result.vectorized = true;
            result.lanes = 8;
            if (result.partial) {
                result.unit = result.lanes;
            }
            return result;
        }

        if (expression.reduction) {
            const auto width = lanes(expression.dtype);
            if (
                !vectorizable ||
                broadcast ||
                !contiguous(expression.reduction->reduced) ||
                expression.reduction->reduced.size < width * 8
            ) {
                return result;
            }

            result.vectorized = true;
            result.lanes = width;
            if (result.partial) {
                result.unit = result.lanes;
            }
            return result;
        }

        if (!broadcast) {
            result.ranged = true;
            result.unit = 1;
            if (!vectorizable) {
                if (packed_math && expression.elements >= 8) {
                    result.vectorized = true;
                    result.math = true;
                    result.lanes = 8;
                }
                return result;
            }
            if (packed_float16 && expression.elements < 64) {
                return result;
            }

            result.vectorized = true;
            result.lanes = packed_float16 ? 8 : 0;
            return result;
        }

        if (
            !span ||
            *span <= 1 ||
            (expression.elements != 0 && expression.elements % *span != 0)
        ) {
            return result;
        }
        if (packed_float16 && (expression.elements < 64 || *span < 8)) {
            return result;
        }
        if (!vectorizable && (!packed_math || *span < 8)) {
            result.ranged = true;
            result.unit = *span;
            return result;
        }

        for (const auto& node : expression.nodes) {
            const auto* input =
                std::get_if<core::runtime::fusion::Input>(&node.instruction);
            if (!input || input->linear || input->axes.empty()) {
                continue;
            }

            bool contiguous = false;
            for (const auto& axis : input->axes) {
                if (axis.divisor == 1) {
                    if (axis.extent != *span || axis.stride != 1) {
                        return result;
                    }
                    contiguous = true;
                } else if (axis.divisor % *span != 0) {
                    return result;
                }
            }
            result.inputs[input->index] = contiguous
                ? Access::span
                : Access::fixed;
        }

        result.vectorized = true;
        result.math = !vectorizable;
        result.ranged = true;
        result.unit = *span;
        result.span = *span;
        result.lanes = packed_float16 || result.math ? 8 : 0;
        return result;
    }
}
