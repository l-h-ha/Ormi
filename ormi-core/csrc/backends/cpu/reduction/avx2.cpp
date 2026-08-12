#include <cstddef>
#include <immintrin.h>

#include "backends/cpu/reduction/avx2.h"

namespace ormi::backends::cpu::avx2 {
    void sum_rows(
        float* sums,
        const float* input,
        std::size_t rows,
        std::size_t columns
    ) {
        std::size_t column = 0;

        for (; column + 32 <= columns; column += 32) {
            auto sum0 = _mm256_setzero_ps();
            auto sum1 = _mm256_setzero_ps();
            auto sum2 = _mm256_setzero_ps();
            auto sum3 = _mm256_setzero_ps();

            for (std::size_t row = 0; row < rows; ++row) {
                const auto* values = input + row * columns + column;
                sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(values));
                sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(values + 8));
                sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(values + 16));
                sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(values + 24));
            }

            _mm256_storeu_ps(sums + column, sum0);
            _mm256_storeu_ps(sums + column + 8, sum1);
            _mm256_storeu_ps(sums + column + 16, sum2);
            _mm256_storeu_ps(sums + column + 24, sum3);
        }

        for (; column + 8 <= columns; column += 8) {
            auto sum = _mm256_setzero_ps();
            for (std::size_t row = 0; row < rows; ++row) {
                sum = _mm256_add_ps(
                    sum,
                    _mm256_loadu_ps(input + row * columns + column)
                );
            }
            _mm256_storeu_ps(sums + column, sum);
        }

        for (; column < columns; ++column) {
            float sum = 0.0f;
            for (std::size_t row = 0; row < rows; ++row) {
                sum += input[row * columns + column];
            }
            sums[column] = sum;
        }
    }
}
