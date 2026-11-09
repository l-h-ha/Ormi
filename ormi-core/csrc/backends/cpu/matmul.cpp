#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/linalg.h"
#include "backends/cpu/matmul/product.h"

namespace ormi::backends::cpu {
    namespace {
        struct Dims {
            std::size_t rows;
            std::size_t inner;
            std::size_t columns;
            std::size_t batch_rank;
            std::size_t batches;
            bool lhs_vector;
            bool rhs_vector;
        };

        Dims dimensions(const core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
            if (lhs.rank() == 0 || rhs.rank() == 0) {
                ORMI_THROW(ShapeError, "Matmul requires tensors with at least one dimension.");
            }

            const bool lhs_vector = lhs.rank() == 1;
            const bool rhs_vector = rhs.rank() == 1;
            const auto lhs_matrix_rank = lhs_vector ? 1u : 2u;
            const auto rhs_matrix_rank = rhs_vector ? 1u : 2u;
            const auto lhs_batch_rank = lhs.rank() - lhs_matrix_rank;
            const auto rhs_batch_rank = rhs.rank() - rhs_matrix_rank;
            const auto batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
            const auto out_matrix_rank =
                static_cast<std::size_t>(!lhs_vector) + static_cast<std::size_t>(!rhs_vector);

            if (out.rank() != batch_rank + out_matrix_rank) {
                ORMI_THROW(ShapeError, "Matmul output rank is invalid.");
            }

            const auto rows = static_cast<std::size_t>(lhs_vector ? 1 : lhs.shape()[lhs.rank() - 2]);
            const auto inner = static_cast<std::size_t>(lhs.shape()[lhs.rank() - 1]);
            const auto rhs_inner = static_cast<std::size_t>(rhs.shape()[rhs.rank() - (rhs_vector ? 1 : 2)]);
            const auto columns = static_cast<std::size_t>(rhs_vector ? 1 : rhs.shape()[rhs.rank() - 1]);

            if (inner != rhs_inner) {
                ORMI_THROW(ShapeError, "Matmul inner dimensions must match.");
            }

            std::size_t batches = 1;
            for (std::size_t axis = 0; axis < batch_rank; ++axis) {
                batches *= static_cast<std::size_t>(out.shape()[axis]);
            }

            return Dims{
                .rows = rows,
                .inner = inner,
                .columns = columns,
                .batch_rank = batch_rank,
                .batches = batches,
                .lhs_vector = lhs_vector,
                .rhs_vector = rhs_vector
            };
        }

        std::size_t batch_offset(
            std::size_t batch,
            const core::Tensor& out,
            std::size_t batch_rank,
            const core::Tensor& tensor,
            std::size_t matrix_rank
        ) {
            const auto tensor_batch_rank = tensor.rank() - matrix_rank;
            const auto rank_offset = batch_rank - tensor_batch_rank;
            const auto strides = tensor.strides();
            std::size_t offset = 0;

            for (std::size_t axis = batch_rank; axis > 0; --axis) {
                const auto output_axis = axis - 1;
                const auto output_dim = static_cast<std::size_t>(out.shape()[output_axis]);
                const auto coordinate = batch % output_dim;
                batch /= output_dim;

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

        template <typename T>
        void matmul_typed(
            core::Tensor& out,
            const core::Tensor& lhs,
            const core::Tensor& rhs,
            const Dims& dims
        ) {
            const auto out_strides = out.strides();
            const auto lhs_strides = lhs.strides();
            const auto rhs_strides = rhs.strides();
            const auto lhs_matrix_rank = dims.lhs_vector ? 1u : 2u;
            const auto rhs_matrix_rank = dims.rhs_vector ? 1u : 2u;
            const auto out_matrix_rank =
                static_cast<std::size_t>(!dims.lhs_vector) + static_cast<std::size_t>(!dims.rhs_vector);
            const auto problem = gemm::Problem{
                .rows = dims.rows,
                .inner = dims.inner,
                .columns = dims.columns,
                .lhs_row_stride = dims.lhs_vector
                                      ? static_cast<std::int64_t>(dims.inner) * lhs_strides[lhs.rank() - 1]
                                      : lhs_strides[lhs.rank() - 2],
                .lhs_column_stride = lhs_strides[lhs.rank() - 1],
                .rhs_row_stride = dims.rhs_vector ? rhs_strides[rhs.rank() - 1] : rhs_strides[rhs.rank() - 2],
                .rhs_column_stride = rhs_strides[rhs.rank() - 1],
                .out_row_stride = dims.lhs_vector ? 0 : out_strides[out.rank() - (dims.rhs_vector ? 1 : 2)],
                .out_column_stride = dims.rhs_vector ? 0 : out_strides[out.rank() - 1]
            };
            auto* out_data = static_cast<T*>(out.data());
            const auto* lhs_data = static_cast<const T*>(lhs.data());
            const auto* rhs_data = static_cast<const T*>(rhs.data());

            for (std::size_t batch = 0; batch < dims.batches; ++batch) {
                const auto out_batch = batch_offset(batch, out, dims.batch_rank, out, out_matrix_rank);
                const auto lhs_batch = batch_offset(batch, out, dims.batch_rank, lhs, lhs_matrix_rank);
                const auto rhs_batch = batch_offset(batch, out, dims.batch_rank, rhs, rhs_matrix_rank);

                gemm::run(out_data + out_batch, lhs_data + lhs_batch, rhs_data + rhs_batch, problem);
            }
        }
    }

    void matmul(core::Tensor& out, const core::Tensor& lhs, const core::Tensor& rhs) {
        if (out.dtype() == core::DType::boolean) {
            ORMI_THROW(DTypeError, "CPU matmul does not support Bool.");
        }
        if (!core::is_numeric(out.dtype())) {
            ORMI_THROW(DTypeError, "CPU matmul cannot handle an unknown dtype.");
        }

        const auto dims = dimensions(out, lhs, rhs);
        if (out.numel() == 0) {
            return;
        }
        if (dims.inner == 0) {
            core::memory::zero(out.data(), out.device(), out.bytes());
            return;
        }

        switch (out.dtype()) {
            case core::DType::float64:
                matmul_typed<double>(out, lhs, rhs, dims);
                return;
            case core::DType::float32:
                matmul_typed<float>(out, lhs, rhs, dims);
                return;
            case core::DType::float16:
                matmul_typed<core::Float16>(out, lhs, rhs, dims);
                return;
            case core::DType::int32:
                matmul_typed<std::int32_t>(out, lhs, rhs, dims);
                return;
            case core::DType::boolean:
                break;
        }

        ORMI_THROW(DTypeError, "CPU matmul cannot handle an unknown dtype.");
    }
}
