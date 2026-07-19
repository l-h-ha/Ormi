#pragma once
#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>

#include <ormi/core/dtype.h>

namespace ormi::backends::cuda::blas {
    struct Matrix {
        std::size_t rows;
        std::size_t columns;
        std::int64_t row_stride;
        std::int64_t column_stride;
    };

    bool gemm(
        core::DType dtype,
        cublasHandle_t handle,
        void* out,
        const void* lhs,
        Matrix lhs_matrix,
        const void* rhs,
        Matrix rhs_matrix,
        int batches,
        long long lhs_batch_stride,
        long long rhs_batch_stride
    );
}
