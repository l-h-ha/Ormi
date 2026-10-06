#pragma once
#include <cstddef>
#include <cstdint>

namespace ormi::backends::cpu::avx2 {
    inline constexpr std::size_t row_gemm_tile = 6;
    inline constexpr std::size_t column_lhs_tile = 6;

    void row_gemv(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::int64_t lhs_row_stride,
        std::int64_t out_row_stride
    );

    void row_gemm(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_row_stride,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::int64_t out_row_stride
    );

    void column_lhs_gemm(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride
    );

    void parallel_row_gemm(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_row_stride,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::int64_t out_row_stride,
        std::size_t work
    );

    void parallel_column_lhs_gemm(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::size_t work
    );
}
