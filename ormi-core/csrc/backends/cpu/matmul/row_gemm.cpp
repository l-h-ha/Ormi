#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "backends/cpu/matmul/avx2.h"

namespace ormi::backends::cpu::avx2 {
    namespace {
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

        void row_gemm_two(
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
                auto* out0 =
                    out + row * static_cast<std::size_t>(out_row_stride);
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
                        const auto* rhs_values = rhs +
                            k * static_cast<std::size_t>(rhs_row_stride) + column;
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
                            rhs + k * static_cast<std::size_t>(rhs_row_stride) +
                                column
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
                        const auto rhs_value = rhs[
                            k * static_cast<std::size_t>(rhs_row_stride) + column
                        ];
                        sum0 += lhs0[lhs_offset] * rhs_value;
                        sum1 += lhs1[lhs_offset] * rhs_value;
                    }
                    out0[column] = sum0;
                    out1[column] = sum1;
                }
            }

            if (row < rows) {
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
        for (; row + row_gemm_tile <= rows; row += row_gemm_tile) {
            auto* out0 = out + row * static_cast<std::size_t>(out_row_stride);
            auto* out1 = out +
                (row + 1) * static_cast<std::size_t>(out_row_stride);
            auto* out2 = out +
                (row + 2) * static_cast<std::size_t>(out_row_stride);
            auto* out3 = out +
                (row + 3) * static_cast<std::size_t>(out_row_stride);
            auto* out4 = out +
                (row + 4) * static_cast<std::size_t>(out_row_stride);
            auto* out5 = out +
                (row + 5) * static_cast<std::size_t>(out_row_stride);
            const auto* lhs0 =
                lhs + row * static_cast<std::size_t>(lhs_row_stride);
            const auto* lhs1 = lhs +
                (row + 1) * static_cast<std::size_t>(lhs_row_stride);
            const auto* lhs2 = lhs +
                (row + 2) * static_cast<std::size_t>(lhs_row_stride);
            const auto* lhs3 = lhs +
                (row + 3) * static_cast<std::size_t>(lhs_row_stride);
            const auto* lhs4 = lhs +
                (row + 4) * static_cast<std::size_t>(lhs_row_stride);
            const auto* lhs5 = lhs +
                (row + 5) * static_cast<std::size_t>(lhs_row_stride);
            std::size_t column = 0;

            // Twelve accumulators leave room for two RHS vectors and one
            // reused LHS broadcast without spilling AVX2 registers.
            for (; column + 16 <= columns; column += 16) {
                auto sum00 = _mm256_setzero_ps();
                auto sum01 = _mm256_setzero_ps();
                auto sum10 = _mm256_setzero_ps();
                auto sum11 = _mm256_setzero_ps();
                auto sum20 = _mm256_setzero_ps();
                auto sum21 = _mm256_setzero_ps();
                auto sum30 = _mm256_setzero_ps();
                auto sum31 = _mm256_setzero_ps();
                auto sum40 = _mm256_setzero_ps();
                auto sum41 = _mm256_setzero_ps();
                auto sum50 = _mm256_setzero_ps();
                auto sum51 = _mm256_setzero_ps();

                for (std::size_t k = 0; k < inner; ++k) {
                    const auto lhs_offset =
                        k * static_cast<std::size_t>(lhs_column_stride);
                    const auto* rhs_values =
                        rhs + k * static_cast<std::size_t>(rhs_row_stride) + column;
                    const auto rhs0 = _mm256_loadu_ps(rhs_values);
                    const auto rhs1 = _mm256_loadu_ps(rhs_values + 8);

                    auto lhs_value = _mm256_set1_ps(lhs0[lhs_offset]);
                    sum00 = _mm256_fmadd_ps(lhs_value, rhs0, sum00);
                    sum01 = _mm256_fmadd_ps(lhs_value, rhs1, sum01);
                    lhs_value = _mm256_set1_ps(lhs1[lhs_offset]);
                    sum10 = _mm256_fmadd_ps(lhs_value, rhs0, sum10);
                    sum11 = _mm256_fmadd_ps(lhs_value, rhs1, sum11);
                    lhs_value = _mm256_set1_ps(lhs2[lhs_offset]);
                    sum20 = _mm256_fmadd_ps(lhs_value, rhs0, sum20);
                    sum21 = _mm256_fmadd_ps(lhs_value, rhs1, sum21);
                    lhs_value = _mm256_set1_ps(lhs3[lhs_offset]);
                    sum30 = _mm256_fmadd_ps(lhs_value, rhs0, sum30);
                    sum31 = _mm256_fmadd_ps(lhs_value, rhs1, sum31);
                    lhs_value = _mm256_set1_ps(lhs4[lhs_offset]);
                    sum40 = _mm256_fmadd_ps(lhs_value, rhs0, sum40);
                    sum41 = _mm256_fmadd_ps(lhs_value, rhs1, sum41);
                    lhs_value = _mm256_set1_ps(lhs5[lhs_offset]);
                    sum50 = _mm256_fmadd_ps(lhs_value, rhs0, sum50);
                    sum51 = _mm256_fmadd_ps(lhs_value, rhs1, sum51);
                }

                _mm256_storeu_ps(out0 + column, sum00);
                _mm256_storeu_ps(out0 + column + 8, sum01);
                _mm256_storeu_ps(out1 + column, sum10);
                _mm256_storeu_ps(out1 + column + 8, sum11);
                _mm256_storeu_ps(out2 + column, sum20);
                _mm256_storeu_ps(out2 + column + 8, sum21);
                _mm256_storeu_ps(out3 + column, sum30);
                _mm256_storeu_ps(out3 + column + 8, sum31);
                _mm256_storeu_ps(out4 + column, sum40);
                _mm256_storeu_ps(out4 + column + 8, sum41);
                _mm256_storeu_ps(out5 + column, sum50);
                _mm256_storeu_ps(out5 + column + 8, sum51);
            }

            for (; column + 8 <= columns; column += 8) {
                auto sum0 = _mm256_setzero_ps();
                auto sum1 = _mm256_setzero_ps();
                auto sum2 = _mm256_setzero_ps();
                auto sum3 = _mm256_setzero_ps();
                auto sum4 = _mm256_setzero_ps();
                auto sum5 = _mm256_setzero_ps();

                for (std::size_t k = 0; k < inner; ++k) {
                    const auto lhs_offset =
                        k * static_cast<std::size_t>(lhs_column_stride);
                    const auto rhs_values = _mm256_loadu_ps(
                        rhs + k * static_cast<std::size_t>(rhs_row_stride) + column
                    );
                    sum0 = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs0[lhs_offset]), rhs_values, sum0
                    );
                    sum1 = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs1[lhs_offset]), rhs_values, sum1
                    );
                    sum2 = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs2[lhs_offset]), rhs_values, sum2
                    );
                    sum3 = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs3[lhs_offset]), rhs_values, sum3
                    );
                    sum4 = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs4[lhs_offset]), rhs_values, sum4
                    );
                    sum5 = _mm256_fmadd_ps(
                        _mm256_set1_ps(lhs5[lhs_offset]), rhs_values, sum5
                    );
                }

                _mm256_storeu_ps(out0 + column, sum0);
                _mm256_storeu_ps(out1 + column, sum1);
                _mm256_storeu_ps(out2 + column, sum2);
                _mm256_storeu_ps(out3 + column, sum3);
                _mm256_storeu_ps(out4 + column, sum4);
                _mm256_storeu_ps(out5 + column, sum5);
            }

            for (; column < columns; ++column) {
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                float sum3 = 0.0f;
                float sum4 = 0.0f;
                float sum5 = 0.0f;

                for (std::size_t k = 0; k < inner; ++k) {
                    const auto lhs_offset =
                        k * static_cast<std::size_t>(lhs_column_stride);
                    const auto rhs_value =
                        rhs[k * static_cast<std::size_t>(rhs_row_stride) + column];
                    sum0 += lhs0[lhs_offset] * rhs_value;
                    sum1 += lhs1[lhs_offset] * rhs_value;
                    sum2 += lhs2[lhs_offset] * rhs_value;
                    sum3 += lhs3[lhs_offset] * rhs_value;
                    sum4 += lhs4[lhs_offset] * rhs_value;
                    sum5 += lhs5[lhs_offset] * rhs_value;
                }

                out0[column] = sum0;
                out1[column] = sum1;
                out2[column] = sum2;
                out3[column] = sum3;
                out4[column] = sum4;
                out5[column] = sum5;
            }
        }

        if (row < rows) {
            row_gemm_two(
                out + row * static_cast<std::size_t>(out_row_stride),
                lhs + row * static_cast<std::size_t>(lhs_row_stride),
                rhs,
                rows - row,
                inner,
                columns,
                lhs_row_stride,
                lhs_column_stride,
                rhs_row_stride,
                out_row_stride
            );
        }
    }
}
