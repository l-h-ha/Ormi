#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <ormi/core/float16.h>

namespace ormi::backends::cpu::gemm::portable {
    template <typename T>
    void row_gemm(
        T* out,
        const T* lhs,
        const T* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_row_stride,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::int64_t out_row_stride
    ) {
        for (std::size_t row = 0; row < rows; ++row) {
            auto* out_row = out + row * static_cast<std::size_t>(
                out_row_stride
            );
            std::fill_n(out_row, columns, T{});

            for (std::size_t k = 0; k < inner; ++k) {
                const auto lhs_value = lhs[
                    row * static_cast<std::size_t>(lhs_row_stride) +
                    k * static_cast<std::size_t>(lhs_column_stride)
                ];
                const auto* rhs_row =
                    rhs + k * static_cast<std::size_t>(rhs_row_stride);

                for (std::size_t column = 0; column < columns; ++column) {
                    out_row[column] += lhs_value * rhs_row[column];
                }
            }
        }
    }

    template <typename T>
    void column_lhs_gemm(
        T* out,
        const T* lhs,
        const T* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride
    ) {
        std::fill_n(out, rows * columns, T{});

        for (std::size_t k = 0; k < inner; ++k) {
            const auto* lhs_column =
                lhs + k * static_cast<std::size_t>(lhs_column_stride);
            const auto* rhs_row =
                rhs + k * static_cast<std::size_t>(rhs_row_stride);

            for (std::size_t row = 0; row < rows; ++row) {
                auto* out_row = out + row * columns;
                const auto lhs_value = lhs_column[row];
                for (std::size_t column = 0; column < columns; ++column) {
                    out_row[column] += lhs_value * rhs_row[column];
                }
            }
        }
    }

    template <typename T>
    void row_gemv(
        T* out,
        const T* lhs,
        const T* rhs,
        std::size_t rows,
        std::size_t inner,
        std::int64_t lhs_row_stride,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::int64_t out_row_stride
    ) {
        for (std::size_t row = 0; row < rows; ++row) {
            T sum{};
            const auto* lhs_row =
                lhs + row * static_cast<std::size_t>(lhs_row_stride);

            for (std::size_t k = 0; k < inner; ++k) {
                sum += lhs_row[
                    k * static_cast<std::size_t>(lhs_column_stride)
                ] * rhs[k * static_cast<std::size_t>(rhs_row_stride)];
            }

            out[row * static_cast<std::size_t>(out_row_stride)] = sum;
        }
    }

    template <typename T>
    void column_lhs_gemv(
        T* out,
        const T* lhs,
        const T* rhs,
        std::size_t rows,
        std::size_t inner,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride
    ) {
        std::fill_n(out, rows, T{});

        for (std::size_t k = 0; k < inner; ++k) {
            const auto* lhs_column =
                lhs + k * static_cast<std::size_t>(lhs_column_stride);
            const auto rhs_value =
                rhs[k * static_cast<std::size_t>(rhs_row_stride)];

            for (std::size_t row = 0; row < rows; ++row) {
                out[row] += lhs_column[row] * rhs_value;
            }
        }
    }

    template <typename T>
    void strided(
        T* out,
        const T* lhs,
        const T* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_row_stride,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::int64_t rhs_column_stride,
        std::int64_t out_row_stride,
        std::int64_t out_column_stride
    ) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const auto out_offset =
                    row * static_cast<std::size_t>(out_row_stride) +
                    column * static_cast<std::size_t>(out_column_stride);
                out[out_offset] = T{};

                for (std::size_t k = 0; k < inner; ++k) {
                    const auto lhs_offset =
                        row * static_cast<std::size_t>(lhs_row_stride) +
                        k * static_cast<std::size_t>(lhs_column_stride);
                    const auto rhs_offset =
                        k * static_cast<std::size_t>(rhs_row_stride) +
                        column * static_cast<std::size_t>(rhs_column_stride);
                    out[out_offset] += lhs[lhs_offset] * rhs[rhs_offset];
                }
            }
        }
    }

    inline void strided(
        core::Float16* out,
        const core::Float16* lhs,
        const core::Float16* rhs,
        std::size_t rows,
        std::size_t inner,
        std::size_t columns,
        std::int64_t lhs_row_stride,
        std::int64_t lhs_column_stride,
        std::int64_t rhs_row_stride,
        std::int64_t rhs_column_stride,
        std::int64_t out_row_stride,
        std::int64_t out_column_stride
    ) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < inner; ++k) {
                    const auto lhs_offset =
                        row * static_cast<std::size_t>(lhs_row_stride) +
                        k * static_cast<std::size_t>(lhs_column_stride);
                    const auto rhs_offset =
                        k * static_cast<std::size_t>(rhs_row_stride) +
                        column * static_cast<std::size_t>(rhs_column_stride);
                    sum += static_cast<float>(lhs[lhs_offset]) *
                        static_cast<float>(rhs[rhs_offset]);
                }

                const auto out_offset =
                    row * static_cast<std::size_t>(out_row_stride) +
                    column * static_cast<std::size_t>(out_column_stride);
                out[out_offset] = core::Float16(sum);
            }
        }
    }
}
