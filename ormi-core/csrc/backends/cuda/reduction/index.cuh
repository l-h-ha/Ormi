#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda::reduction {
    inline constexpr std::size_t max_rank = 64;

    struct Index {
        std::size_t rank = 0;
        std::size_t dims[max_rank]{};
        std::size_t strides[max_rank]{};
        std::uint8_t reduced[max_rank]{};
    };

    static_assert(sizeof(Index) <= 2048);

    inline Index make_index(
        const core::Tensor& input,
        std::span<const std::int64_t> axes
    ) {
        if (input.rank() > max_rank) {
            ORMI_THROW(
                ShapeError,
                "CUDA indexed reduction supports at most 64 dimensions."
            );
        }

        Index index;
        index.rank = input.rank();
        const auto strides = input.strides();

        for (std::size_t axis = 0; axis < index.rank; ++axis) {
            index.dims[axis] = static_cast<std::size_t>(input.shape()[axis]);
            index.strides[axis] = static_cast<std::size_t>(strides[axis]);
        }

        for (const auto axis : axes) {
            index.reduced[static_cast<std::size_t>(axis)] = 1;
        }

        return index;
    }

    __device__ inline std::size_t input_base_offset(
        std::size_t output_linear,
        const Index& index
    ) {
        std::size_t offset = 0;

        for (std::size_t axis = index.rank; axis > 0; --axis) {
            const auto current = axis - 1;

            if (index.reduced[current] != 0) {
                continue;
            }

            const auto coordinate = output_linear % index.dims[current];
            output_linear /= index.dims[current];
            offset += coordinate * index.strides[current];
        }

        return offset;
    }

    __device__ inline std::size_t reduced_offset(
        std::size_t reduced_linear,
        const Index& index
    ) {
        std::size_t offset = 0;

        for (std::size_t axis = index.rank; axis > 0; --axis) {
            const auto current = axis - 1;

            if (index.reduced[current] == 0) {
                continue;
            }

            const auto coordinate = reduced_linear % index.dims[current];
            reduced_linear /= index.dims[current];
            offset += coordinate * index.strides[current];
        }

        return offset;
    }
}
