#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <ormi/core/dtype.h>

#include "backends/cuda/blas/cublas.h"
#include "backends/cuda/check.cuh"

namespace ormi::backends::cuda::blas {
    namespace {
        struct Layout {
            cublasOperation_t operation;
            int leading;
        };

        bool fits_int(std::size_t value) {
            return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
        }

        std::optional<Layout> layout(Matrix matrix) {
            if (matrix.rows == 0 || matrix.columns == 0) {
                return std::nullopt;
            }

            if (
                matrix.column_stride == 1 &&
                matrix.row_stride >= static_cast<std::int64_t>(matrix.columns) &&
                matrix.row_stride <= std::numeric_limits<int>::max()
            ) {
                return Layout{CUBLAS_OP_N, static_cast<int>(matrix.row_stride)};
            }

            if (
                matrix.row_stride == 1 &&
                matrix.column_stride >= static_cast<std::int64_t>(matrix.rows) &&
                matrix.column_stride <= std::numeric_limits<int>::max()
            ) {
                return Layout{CUBLAS_OP_T, static_cast<int>(matrix.column_stride)};
            }

            return std::nullopt;
        }

        bool gemm_float32(
            cublasHandle_t handle,
            float* out,
            const float* lhs,
            Matrix lhs_matrix,
            Layout lhs_layout,
            const float* rhs,
            Matrix rhs_matrix,
            Layout rhs_layout,
            int batches,
            long long lhs_batch_stride,
            long long rhs_batch_stride
        ) {
            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;
            const auto status = batches == 1
                ? cublasSgemm(
                    handle,
                    rhs_layout.operation,
                    lhs_layout.operation,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<int>(lhs_matrix.rows),
                    static_cast<int>(lhs_matrix.columns),
                    &alpha,
                    rhs,
                    rhs_layout.leading,
                    lhs,
                    lhs_layout.leading,
                    &beta,
                    out,
                    static_cast<int>(rhs_matrix.columns)
                )
                : cublasSgemmStridedBatched(
                    handle,
                    rhs_layout.operation,
                    lhs_layout.operation,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<int>(lhs_matrix.rows),
                    static_cast<int>(lhs_matrix.columns),
                    &alpha,
                    rhs,
                    rhs_layout.leading,
                    rhs_batch_stride,
                    lhs,
                    lhs_layout.leading,
                    lhs_batch_stride,
                    &beta,
                    out,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<long long>(lhs_matrix.rows * rhs_matrix.columns),
                    batches
                );
            check(status, batches == 1 ? "cublasSgemm" : "cublasSgemmStridedBatched");
            return true;
        }

        bool gemm_float64(
            cublasHandle_t handle,
            double* out,
            const double* lhs,
            Matrix lhs_matrix,
            Layout lhs_layout,
            const double* rhs,
            Matrix rhs_matrix,
            Layout rhs_layout,
            int batches,
            long long lhs_batch_stride,
            long long rhs_batch_stride
        ) {
            constexpr double alpha = 1.0;
            constexpr double beta = 0.0;
            const auto status = batches == 1
                ? cublasDgemm(
                    handle,
                    rhs_layout.operation,
                    lhs_layout.operation,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<int>(lhs_matrix.rows),
                    static_cast<int>(lhs_matrix.columns),
                    &alpha,
                    rhs,
                    rhs_layout.leading,
                    lhs,
                    lhs_layout.leading,
                    &beta,
                    out,
                    static_cast<int>(rhs_matrix.columns)
                )
                : cublasDgemmStridedBatched(
                    handle,
                    rhs_layout.operation,
                    lhs_layout.operation,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<int>(lhs_matrix.rows),
                    static_cast<int>(lhs_matrix.columns),
                    &alpha,
                    rhs,
                    rhs_layout.leading,
                    rhs_batch_stride,
                    lhs,
                    lhs_layout.leading,
                    lhs_batch_stride,
                    &beta,
                    out,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<long long>(lhs_matrix.rows * rhs_matrix.columns),
                    batches
                );
            check(status, batches == 1 ? "cublasDgemm" : "cublasDgemmStridedBatched");
            return true;
        }

        bool gemm_float16(
            cublasHandle_t handle,
            __half* out,
            const __half* lhs,
            Matrix lhs_matrix,
            Layout lhs_layout,
            const __half* rhs,
            Matrix rhs_matrix,
            Layout rhs_layout,
            int batches,
            long long lhs_batch_stride,
            long long rhs_batch_stride
        ) {
            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;
            const auto status = batches == 1
                ? cublasGemmEx(
                    handle,
                    rhs_layout.operation,
                    lhs_layout.operation,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<int>(lhs_matrix.rows),
                    static_cast<int>(lhs_matrix.columns),
                    &alpha,
                    rhs,
                    CUDA_R_16F,
                    rhs_layout.leading,
                    lhs,
                    CUDA_R_16F,
                    lhs_layout.leading,
                    &beta,
                    out,
                    CUDA_R_16F,
                    static_cast<int>(rhs_matrix.columns),
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                )
                : cublasGemmStridedBatchedEx(
                    handle,
                    rhs_layout.operation,
                    lhs_layout.operation,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<int>(lhs_matrix.rows),
                    static_cast<int>(lhs_matrix.columns),
                    &alpha,
                    rhs,
                    CUDA_R_16F,
                    rhs_layout.leading,
                    rhs_batch_stride,
                    lhs,
                    CUDA_R_16F,
                    lhs_layout.leading,
                    lhs_batch_stride,
                    &beta,
                    out,
                    CUDA_R_16F,
                    static_cast<int>(rhs_matrix.columns),
                    static_cast<long long>(lhs_matrix.rows * rhs_matrix.columns),
                    batches,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
            check(
                status,
                batches == 1 ? "cublasGemmEx" : "cublasGemmStridedBatchedEx"
            );
            return true;
        }
    }

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
    ) {
        if (
            !fits_int(lhs_matrix.rows) ||
            !fits_int(lhs_matrix.columns) ||
            !fits_int(rhs_matrix.columns)
        ) {
            return false;
        }

        const auto lhs_layout = layout(lhs_matrix);
        const auto rhs_layout = layout(rhs_matrix);
        if (!lhs_layout || !rhs_layout) {
            return false;
        }

        switch (dtype) {
            case core::DType::float64:
                return gemm_float64(
                    handle,
                    static_cast<double*>(out),
                    static_cast<const double*>(lhs),
                    lhs_matrix,
                    *lhs_layout,
                    static_cast<const double*>(rhs),
                    rhs_matrix,
                    *rhs_layout,
                    batches,
                    lhs_batch_stride,
                    rhs_batch_stride
                );
            case core::DType::float32:
                return gemm_float32(
                    handle,
                    static_cast<float*>(out),
                    static_cast<const float*>(lhs),
                    lhs_matrix,
                    *lhs_layout,
                    static_cast<const float*>(rhs),
                    rhs_matrix,
                    *rhs_layout,
                    batches,
                    lhs_batch_stride,
                    rhs_batch_stride
                );
            case core::DType::float16:
                return gemm_float16(
                    handle,
                    static_cast<__half*>(out),
                    static_cast<const __half*>(lhs),
                    lhs_matrix,
                    *lhs_layout,
                    static_cast<const __half*>(rhs),
                    rhs_matrix,
                    *rhs_layout,
                    batches,
                    lhs_batch_stride,
                    rhs_batch_stride
                );
            case core::DType::int32:
            case core::DType::boolean:
                return false;
        }

        return false;
    }
}
