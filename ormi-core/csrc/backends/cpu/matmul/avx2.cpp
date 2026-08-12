#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "backends/cpu/matmul/avx2.h"

namespace ormi::backends::cpu::avx2 {
    namespace {
        constexpr std::size_t column_lhs_inner_block = 256;
        constexpr std::size_t column_lhs_block_threshold_bytes = 256 * 1024;

        float horizontal_sum(__m256 value) {
            auto sum = _mm_add_ps(
                _mm256_castps256_ps128(value),
                _mm256_extractf128_ps(value, 1)
            );
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }

        void row_gemm_one(
            float* out,
            const float* lhs,
            const float* rhs,
            std::size_t inner,
            std::size_t columns,
            std::int64_t lhs_column_stride,
            std::int64_t rhs_row_stride
        ) {
            std::size_t column = 0;
            for (; column + 32 <= columns; column += 32) {
                auto sum0 = _mm256_setzero_ps();
                auto sum1 = _mm256_setzero_ps();
                auto sum2 = _mm256_setzero_ps();
                auto sum3 = _mm256_setzero_ps();

                for (std::size_t k = 0; k < inner; ++k) {
                    const auto lhs_value = _mm256_set1_ps(
                        lhs[k * static_cast<std::size_t>(lhs_column_stride)]
                    );
                    const auto* rhs_values =
                        rhs + k * static_cast<std::size_t>(rhs_row_stride) + column;
                    sum0 = _mm256_fmadd_ps(
                        lhs_value, _mm256_loadu_ps(rhs_values), sum0
                    );
                    sum1 = _mm256_fmadd_ps(
                        lhs_value, _mm256_loadu_ps(rhs_values + 8), sum1
                    );
                    sum2 = _mm256_fmadd_ps(
                        lhs_value, _mm256_loadu_ps(rhs_values + 16), sum2
                    );
                    sum3 = _mm256_fmadd_ps(
                        lhs_value, _mm256_loadu_ps(rhs_values + 24), sum3
                    );
                }

                _mm256_storeu_ps(out + column, sum0);
                _mm256_storeu_ps(out + column + 8, sum1);
                _mm256_storeu_ps(out + column + 16, sum2);
                _mm256_storeu_ps(out + column + 24, sum3);
            }

            for (; column + 8 <= columns; column += 8) {
                auto sum = _mm256_setzero_ps();
                for (std::size_t k = 0; k < inner; ++k) {
                    const auto lhs_value = _mm256_set1_ps(
                        lhs[k * static_cast<std::size_t>(lhs_column_stride)]
                    );
                    const auto* rhs_values =
                        rhs + k * static_cast<std::size_t>(rhs_row_stride) + column;
                    sum = _mm256_fmadd_ps(
                        lhs_value, _mm256_loadu_ps(rhs_values), sum
                    );
                }
                _mm256_storeu_ps(out + column, sum);
            }

            for (; column < columns; ++column) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < inner; ++k) {
                    sum +=
                        lhs[k * static_cast<std::size_t>(lhs_column_stride)] *
                        rhs[k * static_cast<std::size_t>(rhs_row_stride) + column];
                }
                out[column] = sum;
            }
        }
    }

    void column_lhs_gemm(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride
    ) {
        if (inner == 0) {
            std::fill_n(out, rows * columns, 0.0f);
            return;
        }

        // Twelve accumulators leave registers for two RHS vectors and one
        // reused LHS broadcast without spilling on AVX2.
        const auto tiled_rows = rows - rows % column_lhs_tile;
        const auto lhs_bytes = rows * inner * sizeof(float);
        const auto inner_block = lhs_bytes >= column_lhs_block_threshold_bytes
            ? column_lhs_inner_block
            : inner;
        // Sweep every output-column panel while one LHS block remains hot.
        for (
            std::size_t k_begin = 0;
            k_begin < inner;
            k_begin += inner_block
        ) {
            const auto k_end = std::min(
                k_begin + inner_block,
                inner
            );
            const bool first_block = k_begin == 0;
            std::size_t column = 0;

            for (; column + 16 <= columns; column += 16) {
                for (
                    std::size_t row = 0;
                    row < tiled_rows;
                    row += column_lhs_tile
                ) {
                    auto* out0 = out + row * columns + column;
                    auto* out1 = out + (row + 1) * columns + column;
                    auto* out2 = out + (row + 2) * columns + column;
                    auto* out3 = out + (row + 3) * columns + column;
                    auto* out4 = out + (row + 4) * columns + column;
                    auto* out5 = out + (row + 5) * columns + column;
                    auto sum00 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out0);
                    auto sum01 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out0 + 8);
                    auto sum10 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out1);
                    auto sum11 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out1 + 8);
                    auto sum20 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out2);
                    auto sum21 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out2 + 8);
                    auto sum30 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out3);
                    auto sum31 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out3 + 8);
                    auto sum40 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out4);
                    auto sum41 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out4 + 8);
                    auto sum50 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out5);
                    auto sum51 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out5 + 8);

                    for (std::size_t k = k_begin; k < k_end; ++k) {
                        const auto* lhs_column = lhs +
                            k * static_cast<std::size_t>(lhs_column_stride);
                        const auto* rhs_row = rhs +
                            k * static_cast<std::size_t>(rhs_row_stride) + column;
                        const auto rhs0 = _mm256_loadu_ps(rhs_row);
                        const auto rhs1 = _mm256_loadu_ps(rhs_row + 8);

                        auto lhs_value = _mm256_set1_ps(lhs_column[row]);
                        sum00 = _mm256_fmadd_ps(lhs_value, rhs0, sum00);
                        sum01 = _mm256_fmadd_ps(lhs_value, rhs1, sum01);
                        lhs_value = _mm256_set1_ps(lhs_column[row + 1]);
                        sum10 = _mm256_fmadd_ps(lhs_value, rhs0, sum10);
                        sum11 = _mm256_fmadd_ps(lhs_value, rhs1, sum11);
                        lhs_value = _mm256_set1_ps(lhs_column[row + 2]);
                        sum20 = _mm256_fmadd_ps(lhs_value, rhs0, sum20);
                        sum21 = _mm256_fmadd_ps(lhs_value, rhs1, sum21);
                        lhs_value = _mm256_set1_ps(lhs_column[row + 3]);
                        sum30 = _mm256_fmadd_ps(lhs_value, rhs0, sum30);
                        sum31 = _mm256_fmadd_ps(lhs_value, rhs1, sum31);
                        lhs_value = _mm256_set1_ps(lhs_column[row + 4]);
                        sum40 = _mm256_fmadd_ps(lhs_value, rhs0, sum40);
                        sum41 = _mm256_fmadd_ps(lhs_value, rhs1, sum41);
                        lhs_value = _mm256_set1_ps(lhs_column[row + 5]);
                        sum50 = _mm256_fmadd_ps(lhs_value, rhs0, sum50);
                        sum51 = _mm256_fmadd_ps(lhs_value, rhs1, sum51);
                    }

                    _mm256_storeu_ps(out0, sum00);
                    _mm256_storeu_ps(out0 + 8, sum01);
                    _mm256_storeu_ps(out1, sum10);
                    _mm256_storeu_ps(out1 + 8, sum11);
                    _mm256_storeu_ps(out2, sum20);
                    _mm256_storeu_ps(out2 + 8, sum21);
                    _mm256_storeu_ps(out3, sum30);
                    _mm256_storeu_ps(out3 + 8, sum31);
                    _mm256_storeu_ps(out4, sum40);
                    _mm256_storeu_ps(out4 + 8, sum41);
                    _mm256_storeu_ps(out5, sum50);
                    _mm256_storeu_ps(out5 + 8, sum51);
                }
            }

            for (; column + 8 <= columns; column += 8) {
                for (
                    std::size_t row = 0;
                    row < tiled_rows;
                    row += column_lhs_tile
                ) {
                    auto* out0 = out + row * columns + column;
                    auto* out1 = out + (row + 1) * columns + column;
                    auto* out2 = out + (row + 2) * columns + column;
                    auto* out3 = out + (row + 3) * columns + column;
                    auto* out4 = out + (row + 4) * columns + column;
                    auto* out5 = out + (row + 5) * columns + column;
                    auto sum0 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out0);
                    auto sum1 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out1);
                    auto sum2 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out2);
                    auto sum3 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out3);
                    auto sum4 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out4);
                    auto sum5 = first_block
                        ? _mm256_setzero_ps()
                        : _mm256_loadu_ps(out5);

                    for (std::size_t k = k_begin; k < k_end; ++k) {
                        const auto* lhs_column = lhs +
                            k * static_cast<std::size_t>(lhs_column_stride);
                        const auto rhs_vector = _mm256_loadu_ps(
                            rhs + k * static_cast<std::size_t>(rhs_row_stride) +
                                column
                        );
                        sum0 = _mm256_fmadd_ps(
                            _mm256_set1_ps(lhs_column[row]), rhs_vector, sum0
                        );
                        sum1 = _mm256_fmadd_ps(
                            _mm256_set1_ps(lhs_column[row + 1]), rhs_vector, sum1
                        );
                        sum2 = _mm256_fmadd_ps(
                            _mm256_set1_ps(lhs_column[row + 2]), rhs_vector, sum2
                        );
                        sum3 = _mm256_fmadd_ps(
                            _mm256_set1_ps(lhs_column[row + 3]), rhs_vector, sum3
                        );
                        sum4 = _mm256_fmadd_ps(
                            _mm256_set1_ps(lhs_column[row + 4]), rhs_vector, sum4
                        );
                        sum5 = _mm256_fmadd_ps(
                            _mm256_set1_ps(lhs_column[row + 5]), rhs_vector, sum5
                        );
                    }

                    _mm256_storeu_ps(out0, sum0);
                    _mm256_storeu_ps(out1, sum1);
                    _mm256_storeu_ps(out2, sum2);
                    _mm256_storeu_ps(out3, sum3);
                    _mm256_storeu_ps(out4, sum4);
                    _mm256_storeu_ps(out5, sum5);
                }
            }

            for (; column < columns; ++column) {
                for (
                    std::size_t row = 0;
                    row < tiled_rows;
                    row += column_lhs_tile
                ) {
                    auto* out0 = out + row * columns + column;
                    auto* out1 = out + (row + 1) * columns + column;
                    auto* out2 = out + (row + 2) * columns + column;
                    auto* out3 = out + (row + 3) * columns + column;
                    auto* out4 = out + (row + 4) * columns + column;
                    auto* out5 = out + (row + 5) * columns + column;
                    float sum0 = first_block ? 0.0f : *out0;
                    float sum1 = first_block ? 0.0f : *out1;
                    float sum2 = first_block ? 0.0f : *out2;
                    float sum3 = first_block ? 0.0f : *out3;
                    float sum4 = first_block ? 0.0f : *out4;
                    float sum5 = first_block ? 0.0f : *out5;

                    for (std::size_t k = k_begin; k < k_end; ++k) {
                        const auto* lhs_column = lhs +
                            k * static_cast<std::size_t>(lhs_column_stride);
                        const auto rhs_value = rhs[
                            k * static_cast<std::size_t>(rhs_row_stride) + column
                        ];
                        sum0 += lhs_column[row] * rhs_value;
                        sum1 += lhs_column[row + 1] * rhs_value;
                        sum2 += lhs_column[row + 2] * rhs_value;
                        sum3 += lhs_column[row + 3] * rhs_value;
                        sum4 += lhs_column[row + 4] * rhs_value;
                        sum5 += lhs_column[row + 5] * rhs_value;
                    }

                    *out0 = sum0;
                    *out1 = sum1;
                    *out2 = sum2;
                    *out3 = sum3;
                    *out4 = sum4;
                    *out5 = sum5;
                }
            }
        }

        if (tiled_rows < rows) {
            row_gemm(
                out + tiled_rows * columns,
                lhs + tiled_rows,
                rhs,
                rows - tiled_rows,
                inner,
                columns,
                1,
                lhs_column_stride,
                rhs_row_stride,
                static_cast<std::int64_t>(columns)
            );
        }
    }

    void row_gemv(
        float* out,
        const float* lhs,
        const float* rhs,
        std::size_t rows,
        std::size_t inner,
        std::int64_t lhs_row_stride,
        std::int64_t out_row_stride
    ) {
        for (std::size_t row = 0; row < rows; ++row) {
            const auto* lhs_row =
                lhs + row * static_cast<std::size_t>(lhs_row_stride);
            auto sum0 = _mm256_setzero_ps();
            auto sum1 = _mm256_setzero_ps();
            auto sum2 = _mm256_setzero_ps();
            auto sum3 = _mm256_setzero_ps();
            std::size_t k = 0;

            for (; k + 32 <= inner; k += 32) {
                sum0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(lhs_row + k),
                    _mm256_loadu_ps(rhs + k),
                    sum0
                );
                sum1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(lhs_row + k + 8),
                    _mm256_loadu_ps(rhs + k + 8),
                    sum1
                );
                sum2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(lhs_row + k + 16),
                    _mm256_loadu_ps(rhs + k + 16),
                    sum2
                );
                sum3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(lhs_row + k + 24),
                    _mm256_loadu_ps(rhs + k + 24),
                    sum3
                );
            }

            sum0 = _mm256_add_ps(sum0, sum2);
            sum1 = _mm256_add_ps(sum1, sum3);
            sum0 = _mm256_add_ps(sum0, sum1);
            for (; k + 8 <= inner; k += 8) {
                sum0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(lhs_row + k),
                    _mm256_loadu_ps(rhs + k),
                    sum0
                );
            }

            auto total = horizontal_sum(sum0);
            for (; k < inner; ++k) {
                total += lhs_row[k] * rhs[k];
            }
            out[row * static_cast<std::size_t>(out_row_stride)] = total;
        }
    }

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
    ) {
        std::size_t row = 0;
        for (; row + 2 <= rows; row += 2) {
            auto* out0 = out + row * static_cast<std::size_t>(out_row_stride);
            auto* out1 = out +
                (row + 1) * static_cast<std::size_t>(out_row_stride);
            const auto* lhs0 =
                lhs + row * static_cast<std::size_t>(lhs_row_stride);
            const auto* lhs1 = lhs +
                (row + 1) * static_cast<std::size_t>(lhs_row_stride);
            std::size_t column = 0;

            for (; column + 32 <= columns; column += 32) {
                auto sum00 = _mm256_setzero_ps();
                auto sum01 = _mm256_setzero_ps();
                auto sum02 = _mm256_setzero_ps();
                auto sum03 = _mm256_setzero_ps();
                auto sum10 = _mm256_setzero_ps();
                auto sum11 = _mm256_setzero_ps();
                auto sum12 = _mm256_setzero_ps();
                auto sum13 = _mm256_setzero_ps();

                for (std::size_t k = 0; k < inner; ++k) {
                    const auto offset =
                        k * static_cast<std::size_t>(lhs_column_stride);
                    const auto lhs_value0 = _mm256_set1_ps(lhs0[offset]);
                    const auto lhs_value1 = _mm256_set1_ps(lhs1[offset]);
                    const auto* rhs_values =
                        rhs + k * static_cast<std::size_t>(rhs_row_stride) + column;
                    const auto rhs0 = _mm256_loadu_ps(rhs_values);
                    const auto rhs1 = _mm256_loadu_ps(rhs_values + 8);
                    const auto rhs2 = _mm256_loadu_ps(rhs_values + 16);
                    const auto rhs3 = _mm256_loadu_ps(rhs_values + 24);
                    sum00 = _mm256_fmadd_ps(lhs_value0, rhs0, sum00);
                    sum01 = _mm256_fmadd_ps(lhs_value0, rhs1, sum01);
                    sum02 = _mm256_fmadd_ps(lhs_value0, rhs2, sum02);
                    sum03 = _mm256_fmadd_ps(lhs_value0, rhs3, sum03);
                    sum10 = _mm256_fmadd_ps(lhs_value1, rhs0, sum10);
                    sum11 = _mm256_fmadd_ps(lhs_value1, rhs1, sum11);
                    sum12 = _mm256_fmadd_ps(lhs_value1, rhs2, sum12);
                    sum13 = _mm256_fmadd_ps(lhs_value1, rhs3, sum13);
                }

                _mm256_storeu_ps(out0 + column, sum00);
                _mm256_storeu_ps(out0 + column + 8, sum01);
                _mm256_storeu_ps(out0 + column + 16, sum02);
                _mm256_storeu_ps(out0 + column + 24, sum03);
                _mm256_storeu_ps(out1 + column, sum10);
                _mm256_storeu_ps(out1 + column + 8, sum11);
                _mm256_storeu_ps(out1 + column + 16, sum12);
                _mm256_storeu_ps(out1 + column + 24, sum13);
            }

            for (; column + 8 <= columns; column += 8) {
                auto sum0 = _mm256_setzero_ps();
                auto sum1 = _mm256_setzero_ps();
                for (std::size_t k = 0; k < inner; ++k) {
                    const auto offset =
                        k * static_cast<std::size_t>(lhs_column_stride);
                    const auto rhs_values = _mm256_loadu_ps(
                        rhs + k * static_cast<std::size_t>(rhs_row_stride) + column
                    );
                    sum0 = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs0[offset]), rhs_values, sum0
                    );
                    sum1 = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs1[offset]), rhs_values, sum1
                    );
                }
                _mm256_storeu_ps(out0 + column, sum0);
                _mm256_storeu_ps(out1 + column, sum1);
            }

            for (; column < columns; ++column) {
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                for (std::size_t k = 0; k < inner; ++k) {
                    const auto lhs_offset =
                        k * static_cast<std::size_t>(lhs_column_stride);
                    const auto rhs_value =
                        rhs[k * static_cast<std::size_t>(rhs_row_stride) + column];
                    sum0 += lhs0[lhs_offset] * rhs_value;
                    sum1 += lhs1[lhs_offset] * rhs_value;
                }
                out0[column] = sum0;
                out1[column] = sum1;
            }
        }

        if (row == rows) {
            return;
        }

        row_gemm_one(
            out + row * static_cast<std::size_t>(out_row_stride),
            lhs + row * static_cast<std::size_t>(lhs_row_stride),
            rhs,
            inner,
            columns,
            lhs_column_stride,
            rhs_row_stride
        );
    }

}
