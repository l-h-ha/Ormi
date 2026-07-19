#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/linalg.h"
#include "backends/cpu/blas/openblas.h"

namespace ormi::backends::cpu {
    namespace {
        std::size_t batch_offset(
            std::size_t batch,
            const core::Shape& batch_shape,
            const core::Tensor& tensor,
            std::size_t matrix_rank
        ) {
            const auto output_batch_rank = batch_shape.rank();
            const auto tensor_batch_rank = tensor.rank() - matrix_rank;
            const auto rank_offset = output_batch_rank - tensor_batch_rank;
            const auto strides = tensor.strides();
            std::size_t offset = 0;

            for (std::size_t axis = output_batch_rank; axis > 0; --axis) {
                const auto output_axis = axis - 1;
                const auto output_dim = static_cast<std::size_t>(batch_shape[output_axis]);
                const auto coordinate = batch % output_dim;
                batch /= output_dim;

                if (output_axis >= rank_offset) {
                    const auto tensor_axis = output_axis - rank_offset;
                    if (tensor.shape()[tensor_axis] != 1) {
                        offset += coordinate * static_cast<std::size_t>(strides[tensor_axis]);
                    }
                }
            }

            return offset;
        }

        template <typename T>
        void matmul_typed(core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
            const auto shape = core::ops::matmul_shape(lhs.shape(), rhs.shape());
            if (out.numel() == 0) {
                return;
            }

            auto* out_data = static_cast<T*>(out.data());
            if (shape.inner == 0) {
                std::fill_n(out_data, out.numel(), T{});
                return;
            }

            const auto* lhs_data = static_cast<const T*>(lhs.data());
            const auto* rhs_data = static_cast<const T*>(rhs.data());
            const auto batches = shape.batch.numel();
            const auto out_strides = out.strides();
            const auto lhs_strides = lhs.strides();
            const auto rhs_strides = rhs.strides();
            const auto lhs_matrix_rank = shape.lhs_vector ? 1u : 2u;
            const auto rhs_matrix_rank = shape.rhs_vector ? 1u : 2u;
            const auto out_matrix_rank =
                static_cast<std::size_t>(!shape.lhs_vector) +
                static_cast<std::size_t>(!shape.rhs_vector);
            const auto lhs_row_stride = shape.lhs_vector
                ? static_cast<std::int64_t>(shape.inner) * lhs_strides[lhs.rank() - 1]
                : lhs_strides[lhs.rank() - 2];
            const auto lhs_column_stride = lhs_strides[lhs.rank() - 1];
            const auto rhs_row_stride = shape.rhs_vector
                ? rhs_strides[rhs.rank() - 1]
                : rhs_strides[rhs.rank() - 2];
            const auto rhs_column_stride = rhs_strides[rhs.rank() - 1];
            const auto out_row_stride = shape.lhs_vector
                ? 0
                : out_strides[out.rank() - (shape.rhs_vector ? 1 : 2)];
            const auto out_column_stride = shape.rhs_vector
                ? 0
                : out_strides[out.rank() - 1];

            for (std::size_t batch = 0; batch < batches; ++batch) {
                const auto out_batch = batch_offset(batch, shape.batch, out, out_matrix_rank);
                const auto lhs_batch = batch_offset(batch, shape.batch, lhs, lhs_matrix_rank);
                const auto rhs_batch = batch_offset(batch, shape.batch, rhs, rhs_matrix_rank);

                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    const auto lhs_matrix = blas::Matrix{
                        shape.rows,
                        shape.inner,
                        lhs_row_stride,
                        lhs_column_stride
                    };
                    const auto rhs_matrix = blas::Matrix{
                        shape.inner,
                        shape.columns,
                        rhs_row_stride,
                        rhs_column_stride
                    };

                    if (blas::gemm(
                        out_data + out_batch,
                        lhs_data + lhs_batch,
                        lhs_matrix,
                        rhs_data + rhs_batch,
                        rhs_matrix
                    )) {
                        continue;
                    }
                }

                for (std::size_t row = 0; row < shape.rows; ++row) {
                    for (std::size_t column = 0; column < shape.columns; ++column) {
                        const auto out_offset =
                            out_batch +
                            row * static_cast<std::size_t>(out_row_stride) +
                            column * static_cast<std::size_t>(out_column_stride);
                        out_data[out_offset] = T{};
                    }

                    for (std::size_t k = 0; k < shape.inner; ++k) {
                        const auto lhs_offset =
                            lhs_batch +
                            row * static_cast<std::size_t>(lhs_row_stride) +
                            k * static_cast<std::size_t>(lhs_column_stride);
                        const auto lhs_value = lhs_data[lhs_offset];

                        for (std::size_t column = 0; column < shape.columns; ++column) {
                            const auto rhs_offset =
                                rhs_batch +
                                k * static_cast<std::size_t>(rhs_row_stride) +
                                column * static_cast<std::size_t>(rhs_column_stride);
                            const auto out_offset =
                                out_batch +
                                row * static_cast<std::size_t>(out_row_stride) +
                                column * static_cast<std::size_t>(out_column_stride);
                            out_data[out_offset] += lhs_value * rhs_data[rhs_offset];
                        }
                    }
                }
            }
        }

        void matmul_float16(core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
            const auto shape = core::ops::matmul_shape(lhs.shape(), rhs.shape());
            if (out.numel() == 0) {
                return;
            }

            auto* out_data = static_cast<core::Float16*>(out.data());
            if (shape.inner == 0) {
                std::fill_n(out_data, out.numel(), core::Float16{});
                return;
            }

            const auto* lhs_data = static_cast<const core::Float16*>(lhs.data());
            const auto* rhs_data = static_cast<const core::Float16*>(rhs.data());
            const auto batches = shape.batch.numel();
            const auto out_strides = out.strides();
            const auto lhs_strides = lhs.strides();
            const auto rhs_strides = rhs.strides();
            const auto lhs_matrix_rank = shape.lhs_vector ? 1u : 2u;
            const auto rhs_matrix_rank = shape.rhs_vector ? 1u : 2u;
            const auto out_matrix_rank =
                static_cast<std::size_t>(!shape.lhs_vector) +
                static_cast<std::size_t>(!shape.rhs_vector);
            const auto lhs_row_stride = shape.lhs_vector
                ? static_cast<std::int64_t>(shape.inner) * lhs_strides[lhs.rank() - 1]
                : lhs_strides[lhs.rank() - 2];
            const auto lhs_column_stride = lhs_strides[lhs.rank() - 1];
            const auto rhs_row_stride = shape.rhs_vector
                ? rhs_strides[rhs.rank() - 1]
                : rhs_strides[rhs.rank() - 2];
            const auto rhs_column_stride = rhs_strides[rhs.rank() - 1];
            const auto out_row_stride = shape.lhs_vector
                ? 0
                : out_strides[out.rank() - (shape.rhs_vector ? 1 : 2)];
            const auto out_column_stride = shape.rhs_vector
                ? 0
                : out_strides[out.rank() - 1];

            for (std::size_t batch = 0; batch < batches; ++batch) {
                const auto out_batch = batch_offset(batch, shape.batch, out, out_matrix_rank);
                const auto lhs_batch = batch_offset(batch, shape.batch, lhs, lhs_matrix_rank);
                const auto rhs_batch = batch_offset(batch, shape.batch, rhs, rhs_matrix_rank);

                for (std::size_t row = 0; row < shape.rows; ++row) {
                    for (std::size_t column = 0; column < shape.columns; ++column) {
                        float sum = 0.0f;

                        for (std::size_t k = 0; k < shape.inner; ++k) {
                            const auto lhs_offset = lhs_batch +
                                row * static_cast<std::size_t>(lhs_row_stride) +
                                k * static_cast<std::size_t>(lhs_column_stride);
                            const auto rhs_offset = rhs_batch +
                                k * static_cast<std::size_t>(rhs_row_stride) +
                                column * static_cast<std::size_t>(rhs_column_stride);
                            sum += static_cast<float>(lhs_data[lhs_offset]) *
                                static_cast<float>(rhs_data[rhs_offset]);
                        }

                        const auto out_offset = out_batch +
                            row * static_cast<std::size_t>(out_row_stride) +
                            column * static_cast<std::size_t>(out_column_stride);
                        out_data[out_offset] = core::Float16(sum);
                    }
                }
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
                matmul_float16(out, lhs, rhs);
                return;
            case core::DType::int32:
                matmul_typed<std::int32_t>(out, lhs, rhs);
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CPU matmul does not support Bool.");
        }

        ORMI_THROW(DTypeError, "CPU matmul cannot handle an unknown dtype.");
    }
}
