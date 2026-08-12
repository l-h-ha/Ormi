#pragma once
#include <cstddef>
#include <cstdint>

namespace ormi::backends::cpu::blas {
    struct Matrix {
        std::size_t rows;
        std::size_t columns;
        std::int64_t row_stride;
        std::int64_t column_stride;
    };

    bool available();
    bool generic();
    bool gemm(float* out, const float* lhs, Matrix lhs_matrix, const float* rhs, Matrix rhs_matrix);
    bool gemm(double* out, const double* lhs, Matrix lhs_matrix, const double* rhs, Matrix rhs_matrix);
}
