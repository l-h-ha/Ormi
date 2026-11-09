#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <ormi/backend/fusion.h>
#include <ormi/core/shape.h>

namespace ormi::core::runtime::fusion::layout {
    namespace detail {
        inline Iteration iteration(
            const Shape& shape,
            std::span<const std::int64_t> reduced_axes,
            bool reduced
        ) {
            std::vector<std::size_t> selected;
            selected.reserve(shape.rank());

            for (std::size_t axis = 0; axis < shape.rank(); ++axis) {
                const auto is_reduced = std::binary_search(
                    reduced_axes.begin(),
                    reduced_axes.end(),
                    static_cast<std::int64_t>(axis)
                );
                if (is_reduced == reduced) {
                    selected.push_back(axis);
                }
            }

            std::size_t size = 1;
            for (const auto axis : selected) {
                size *= static_cast<std::size_t>(shape[axis]);
            }

            if (size == 0) {
                return Iteration{0, {}};
            }

            const auto source_strides = contiguous_strides(shape);
            std::vector<Axis> axes(selected.size());
            std::size_t divisor = 1;

            for (std::size_t i = selected.size(); i > 0; --i) {
                const auto axis = selected[i - 1];
                const auto extent = static_cast<std::size_t>(shape[axis]);
                axes[i - 1] = Axis{
                    divisor,
                    extent,
                    static_cast<std::size_t>(source_strides[axis])
                };
                divisor *= extent;
            }
            return Iteration{size, std::move(axes)};
        }
    }

    inline std::vector<std::size_t> dimensions(const Shape& shape) {
        std::vector<std::size_t> result;
        result.reserve(shape.rank());
        for (std::size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            result.push_back(static_cast<std::size_t>(shape[dimension]));
        }
        return result;
    }

    inline bool input_axes(
        const Shape& shape,
        std::span<const std::int64_t> strides,
        const Shape& output,
        std::vector<Axis>& axes
    ) {
        axes.clear();
        if (shape.rank() > output.rank()) {
            return false;
        }

        if (output.empty()) {
            return true;
        }

        const auto output_strides = contiguous_strides(output);
        const auto rank_offset = output.rank() - shape.rank();
        axes.reserve(shape.rank());

        for (std::size_t axis = rank_offset; axis < output.rank(); ++axis) {
            const auto input_axis = axis - rank_offset;
            const auto input_extent = shape[input_axis];
            const auto output_extent = output[axis];

            if (input_extent != 1 && input_extent != output_extent) {
                return false;
            }

            if (input_extent != 1 && strides[input_axis] != 0) {
                axes.push_back(Axis{
                    static_cast<std::size_t>(output_strides[axis]),
                    static_cast<std::size_t>(output_extent),
                    static_cast<std::size_t>(strides[input_axis])
                });
            }
        }
        return true;
    }

    // Reduction axes are already normalized by the operation.
    inline Reduction reduction(const Shape& shape, std::span<const std::int64_t> axes, ops::ReduceOp op) {
        return Reduction{
            op,
            detail::iteration(shape, axes, false),
            detail::iteration(shape, axes, true)
        };
    }
}
