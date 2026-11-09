#pragma once
#include <algorithm>
#include <cstddef>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda::elementwise {
    inline constexpr std::size_t max_rank = 64;
    inline constexpr unsigned int threads = 256;
    inline constexpr std::size_t max_blocks = 65535;

    struct Index {
        std::size_t rank = 0;
        std::size_t dims[max_rank]{};
        std::size_t strides[max_rank]{};
    };

    struct BinaryIndex {
        std::size_t rank = 0;
        std::size_t dims[max_rank]{};
        std::size_t lhs_strides[max_rank]{};
        std::size_t rhs_strides[max_rank]{};
    };

    static_assert(sizeof(Index) <= 2048);
    static_assert(sizeof(BinaryIndex) <= 2048);

    inline void require_supported_rank(std::size_t rank) {
        if (rank > max_rank) {
            ORMI_THROW(ShapeError, "CUDA strided elementwise indexing supports at most 64 dimensions.");
        }
    }

    inline Index make_index(const core::Tensor& input) {
        require_supported_rank(input.rank());

        Index index;
        index.rank = input.rank();
        const auto strides = input.strides();

        for (std::size_t axis = 0; axis < index.rank; ++axis) {
            index.dims[axis] = static_cast<std::size_t>(input.shape()[axis]);
            index.strides[axis] = static_cast<std::size_t>(strides[axis]);
        }

        return index;
    }

    inline BinaryIndex make_index(const core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
        require_supported_rank(out.rank());

        BinaryIndex index;
        index.rank = out.rank();
        const auto lhs_rank_offset = index.rank - lhs.rank();
        const auto rhs_rank_offset = index.rank - rhs.rank();
        const auto lhs_strides = lhs.strides();
        const auto rhs_strides = rhs.strides();

        for (std::size_t axis = 0; axis < index.rank; ++axis) {
            index.dims[axis] = static_cast<std::size_t>(out.shape()[axis]);

            if (axis >= lhs_rank_offset) {
                const auto lhs_axis = axis - lhs_rank_offset;
                if (lhs.shape()[lhs_axis] != 1) {
                    index.lhs_strides[axis] = static_cast<std::size_t>(lhs_strides[lhs_axis]);
                }
            }

            if (axis >= rhs_rank_offset) {
                const auto rhs_axis = axis - rhs_rank_offset;
                if (rhs.shape()[rhs_axis] != 1) {
                    index.rhs_strides[axis] = static_cast<std::size_t>(rhs_strides[rhs_axis]);
                }
            }
        }

        return index;
    }

    inline unsigned int blocks(std::size_t elements) {
        const auto required = elements / threads + (elements % threads != 0);
        return static_cast<unsigned int>(std::min(required, max_blocks));
    }

    __device__ inline std::size_t storage_offset(std::size_t linear, const Index& index) {
        std::size_t offset = 0;

        for (std::size_t axis = index.rank; axis > 0; --axis) {
            const auto current = axis - 1;
            const auto coordinate = linear % index.dims[current];
            linear /= index.dims[current];
            offset += coordinate * index.strides[current];
        }

        return offset;
    }

    __device__ inline void storage_offsets(
        std::size_t linear,
        const BinaryIndex& index,
        std::size_t& lhs_offset,
        std::size_t& rhs_offset
    ) {
        lhs_offset = 0;
        rhs_offset = 0;

        for (std::size_t axis = index.rank; axis > 0; --axis) {
            const auto current = axis - 1;
            const auto coordinate = linear % index.dims[current];
            linear /= index.dims[current];
            lhs_offset += coordinate * index.lhs_strides[current];
            rhs_offset += coordinate * index.rhs_strides[current];
        }
    }
}
