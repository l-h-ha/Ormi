#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/blas/cublas.h"
#include "backends/cuda/check.cuh"
#include "backends/cuda/linalg.h"
#include "backends/cuda/memory.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        constexpr unsigned int tile = 16;
        constexpr std::size_t max_grid = 65535;

        std::size_t batch_offset(
            std::size_t batch,
            const core::Shape& batch_shape,
            const core::Tensor& tensor,
            std::size_t matrix_rank
        ) {
            const auto output_rank = batch_shape.rank();
            const auto tensor_rank = tensor.rank() - matrix_rank;
            const auto rank_offset = output_rank - tensor_rank;
            const auto strides = tensor.strides();
            std::size_t offset = 0;

            for (std::size_t axis = output_rank; axis > 0; --axis) {
                const auto output_axis = axis - 1;
                const auto dim = static_cast<std::size_t>(batch_shape[output_axis]);
                const auto coordinate = batch % dim;
                batch /= dim;

                if (output_axis < rank_offset) {
                    continue;
                }

                const auto tensor_axis = output_axis - rank_offset;
                if (tensor.shape()[tensor_axis] == 1) {
                    continue;
                }

                offset += coordinate * static_cast<std::size_t>(strides[tensor_axis]);
            }

            return offset;
        }

        bool fits_int(std::size_t value) {
            return value <= static_cast<std::size_t>(std::numeric_limits<int>::max());
        }

        std::optional<long long> regular_batch_stride(
            const core::Shape& batch_shape,
            const core::Tensor& tensor,
            std::size_t matrix_rank
        ) {
            const auto output_rank = batch_shape.rank();
            const auto tensor_rank = tensor.rank() - matrix_rank;
            const auto rank_offset = output_rank - tensor_rank;
            const auto strides = tensor.strides();
            std::optional<long long> result;
            std::size_t trailing = 1;

            for (std::size_t axis = output_rank; axis > 0; --axis) {
                const auto output_axis = axis - 1;
                const auto output_dim = batch_shape[output_axis];

                if (output_dim != 1) {
                    std::size_t stride = 0;
                    if (output_axis >= rank_offset) {
                        const auto tensor_axis = output_axis - rank_offset;
                        if (tensor.shape()[tensor_axis] != 1) {
                            stride = static_cast<std::size_t>(strides[tensor_axis]);
                        }
                    }

                    if (stride % trailing != 0) {
                        return std::nullopt;
                    }

                    const auto candidate = stride / trailing;
                    if (candidate > static_cast<std::size_t>(std::numeric_limits<long long>::max())) {
                        return std::nullopt;
                    }
                    const auto candidate_stride = static_cast<long long>(candidate);
                    if (result && *result != candidate_stride) {
                        return std::nullopt;
                    }
                    result = candidate_stride;
                }

                trailing *= static_cast<std::size_t>(output_dim);
            }

            return result.value_or(0);
        }

        template <typename T>
        struct Compute {
            using Type = T;

            __device__ static Type load(T value) {
                return value;
            }

            __device__ static T store(Type value) {
                return value;
            }
        };

        template <>
        struct Compute<__half> {
            using Type = float;

            __device__ static Type load(__half value) {
                return __half2float(value);
            }

            __device__ static __half store(Type value) {
                return __float2half(value);
            }
        };

        template <typename T>
        __global__ void matmul_kernel(
            T* out,
            const T* lhs,
            const T* rhs,
            std::size_t rows,
            std::size_t inner,
            std::size_t columns,
            std::int64_t lhs_row_stride,
            std::int64_t lhs_column_stride,
            std::int64_t rhs_row_stride,
            std::int64_t rhs_column_stride
        ) {
            for (std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
                 row < rows;
                 row += static_cast<std::size_t>(blockDim.y) * gridDim.y) {
                for (std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                     column < columns;
                     column += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                    typename Compute<T>::Type sum{};

                    for (std::size_t k = 0; k < inner; ++k) {
                        const auto lhs_value =
                            Compute<T>::load(lhs[row * lhs_row_stride + k * lhs_column_stride]);
                        const auto rhs_value =
                            Compute<T>::load(rhs[k * rhs_row_stride + column * rhs_column_stride]);
                        sum += lhs_value * rhs_value;
                    }

                    out[row * columns + column] = Compute<T>::store(sum);
                }
            }
        }

        template <typename T>
        void launch_fallback(
            T* out,
            const T* lhs,
            const T* rhs,
            const core::ops::MatmulShape& shape,
            blas::Matrix lhs_matrix,
            blas::Matrix rhs_matrix,
            cudaStream_t stream
        ) {
            if (shape.rows == 0 || shape.columns == 0) {
                return;
            }

            const auto x_blocks = std::min((shape.columns + tile - 1) / tile, max_grid);
            const auto y_blocks = std::min((shape.rows + tile - 1) / tile, max_grid);
            const dim3 threads(tile, tile);
            const dim3 blocks(static_cast<unsigned int>(x_blocks), static_cast<unsigned int>(y_blocks));

            matmul_kernel<<<blocks, threads, 0, stream>>>(
                out,
                lhs,
                rhs,
                shape.rows,
                shape.inner,
                shape.columns,
                lhs_matrix.row_stride,
                lhs_matrix.column_stride,
                rhs_matrix.row_stride,
                rhs_matrix.column_stride
            );
            check(cudaGetLastError(), "matmul_kernel");
        }

        template <typename T>
        void matmul_typed(core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
            auto& cuda_state = state(out.device());
            check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

            const auto shape = core::ops::matmul_shape(lhs.shape(), rhs.shape());
            if (out.numel() == 0) {
                return;
            }

            if (shape.inner == 0) {
                fill(out.device(), out.data(), 0, out.bytes());
                return;
            }

            auto* out_data = static_cast<T*>(out.data());
            const auto* lhs_data = static_cast<const T*>(lhs.data());
            const auto* rhs_data = static_cast<const T*>(rhs.data());
            const auto lhs_strides = lhs.strides();
            const auto rhs_strides = rhs.strides();
            const auto lhs_rank = shape.lhs_vector ? 1u : 2u;
            const auto rhs_rank = shape.rhs_vector ? 1u : 2u;
            const blas::Matrix lhs_matrix{
                .rows = shape.rows,
                .columns = shape.inner,
                .row_stride = shape.lhs_vector ? static_cast<std::int64_t>(shape.inner) * lhs_strides.back()
                                               : lhs_strides[lhs.rank() - 2],
                .column_stride = lhs_strides.back()
            };
            const blas::Matrix rhs_matrix{
                .rows = shape.inner,
                .columns = shape.columns,
                .row_stride = shape.rhs_vector ? rhs_strides.back() : rhs_strides[rhs.rank() - 2],
                .column_stride = rhs_strides.back()
            };
            const auto matrix_elements = shape.rows * shape.columns;
            const auto batches = shape.batch.numel();

            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, __half>) {
                const auto lhs_batch_stride = regular_batch_stride(shape.batch, lhs, lhs_rank);
                const auto rhs_batch_stride = regular_batch_stride(shape.batch, rhs, rhs_rank);
                if (batches > 0 && fits_int(batches) && lhs_batch_stride && rhs_batch_stride &&
                    blas::gemm(
                        out.dtype(),
                        cuda_state.cublas(),
                        out_data,
                        lhs_data,
                        lhs_matrix,
                        rhs_data,
                        rhs_matrix,
                        static_cast<int>(batches),
                        *lhs_batch_stride,
                        *rhs_batch_stride
                    )) {
                    return;
                }
            }

            std::size_t batch = 0;
            while (batch < batches) {
                const auto lhs_offset = batch_offset(batch, shape.batch, lhs, lhs_rank);
                const auto rhs_offset = batch_offset(batch, shape.batch, rhs, rhs_rank);
                auto* out_batch = out_data + batch * matrix_elements;
                const auto* lhs_batch = lhs_data + lhs_offset;
                const auto* rhs_batch = rhs_data + rhs_offset;

                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                              std::is_same_v<T, __half>) {
                    std::size_t run = 1;
                    long long lhs_stride = 0;
                    long long rhs_stride = 0;

                    if (batch + 1 < batches) {
                        const auto next_lhs = batch_offset(batch + 1, shape.batch, lhs, lhs_rank);
                        const auto next_rhs = batch_offset(batch + 1, shape.batch, rhs, rhs_rank);

                        if (next_lhs >= lhs_offset && next_rhs >= rhs_offset) {
                            const auto lhs_delta = next_lhs - lhs_offset;
                            const auto rhs_delta = next_rhs - rhs_offset;
                            const auto stride_limit =
                                static_cast<std::size_t>(std::numeric_limits<long long>::max());

                            if (lhs_delta <= stride_limit && rhs_delta <= stride_limit) {
                                lhs_stride = static_cast<long long>(lhs_delta);
                                rhs_stride = static_cast<long long>(rhs_delta);
                                run = 2;

                                while (batch + run < batches &&
                                       run < static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                                    const auto run_lhs =
                                        batch_offset(batch + run, shape.batch, lhs, lhs_rank);
                                    const auto run_rhs =
                                        batch_offset(batch + run, shape.batch, rhs, rhs_rank);
                                    const auto expected_lhs =
                                        lhs_offset + run * static_cast<std::size_t>(lhs_stride);
                                    const auto expected_rhs =
                                        rhs_offset + run * static_cast<std::size_t>(rhs_stride);

                                    if (run_lhs != expected_lhs || run_rhs != expected_rhs) {
                                        break;
                                    }

                                    ++run;
                                }
                            }
                        }
                    }

                    if (blas::gemm(
                            out.dtype(),
                            cuda_state.cublas(),
                            out_batch,
                            lhs_batch,
                            lhs_matrix,
                            rhs_batch,
                            rhs_matrix,
                            static_cast<int>(run),
                            lhs_stride,
                            rhs_stride
                        )) {
                        batch += run;
                        continue;
                    }
                }

                launch_fallback(
                    out_batch,
                    lhs_batch,
                    rhs_batch,
                    shape,
                    lhs_matrix,
                    rhs_matrix,
                    cuda_state.stream()
                );
                ++batch;
            }
        }
    }

    void matmul(core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
        switch (out.dtype()) {
            case core::DType::float64:
                matmul_typed<double>(out, lhs, rhs);
                return;
            case core::DType::float32:
                matmul_typed<float>(out, lhs, rhs);
                return;
            case core::DType::float16:
                matmul_typed<__half>(out, lhs, rhs);
                return;
            case core::DType::int32:
                matmul_typed<std::int32_t>(out, lhs, rhs);
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CUDA matmul does not support Bool.");
        }

        ORMI_THROW(DTypeError, "CUDA matmul cannot handle an unknown dtype.");
    }
}
