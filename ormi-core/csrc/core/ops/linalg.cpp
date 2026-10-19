#include <memory>
#include <utility>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/linalg.h"
#include "core/precision.h"

namespace ormi::core::ops {
    namespace {
        void compute_matmul(Tensor& out) {
            runtime::linalg::matmul(out, *out.parents()[0], *out.parents()[1]);
        }

        std::vector<std::int64_t> swap_last_axes(std::size_t rank) {
            std::vector<std::int64_t> axes(rank);

            for (std::size_t axis = 0; axis < rank; ++axis) {
                axes[axis] = static_cast<std::int64_t>(axis);
            }

            std::swap(axes[rank - 2], axes[rank - 1]);
            return axes;
        }

        std::shared_ptr<Tensor> lhs_matrix(const std::shared_ptr<Tensor>& tensor) {
            if (tensor->rank() != 1) {
                return tensor;
            }

            return reshape(tensor, Shape{1, tensor->shape()[0]});
        }

        std::shared_ptr<Tensor> rhs_matrix(const std::shared_ptr<Tensor>& tensor) {
            if (tensor->rank() != 1) {
                return tensor;
            }

            return reshape(tensor, Shape{tensor->shape()[0], 1});
        }

        Shape matrix_output_shape(const MatmulShape& shape) {
            auto dims = shape.batch.dims();
            dims.push_back(static_cast<std::int64_t>(shape.rows));
            dims.push_back(static_cast<std::int64_t>(shape.columns));
            return Shape(std::move(dims));
        }

        std::shared_ptr<Tensor> restore_vector_gradient(
            const std::shared_ptr<Tensor>& gradient,
            const std::shared_ptr<Tensor>& input
        ) {
            return input->rank() == 1
                ? reshape(gradient, input->shape())
                : gradient;
        }

        void adjoint_matmul(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& lhs = node->parents()[0];
            const auto& rhs = node->parents()[1];
            const auto shape = matmul_shape(lhs->shape(), rhs->shape());
            const auto matrix_lhs = lhs_matrix(lhs);
            const auto matrix_rhs = rhs_matrix(rhs);
            const auto matrix_upstream = shape.lhs_vector || shape.rhs_vector
                ? reshape(upstream, matrix_output_shape(shape))
                : upstream;
            const auto rhs_transposed = transpose(
                matrix_rhs,
                swap_last_axes(matrix_rhs->rank())
            );
            const auto lhs_transposed = transpose(
                matrix_lhs,
                swap_last_axes(matrix_lhs->rank())
            );
            const auto lhs_gradient = unbroadcast(
                matmul(matrix_upstream, rhs_transposed),
                matrix_lhs->shape()
            );
            const auto rhs_gradient = unbroadcast(
                matmul(lhs_transposed, matrix_upstream),
                matrix_rhs->shape()
            );
            gradients.add(lhs, restore_vector_gradient(lhs_gradient, lhs));
            gradients.add(rhs, restore_vector_gradient(rhs_gradient, rhs));
        }

        const Op matmul_op{
            .name = "matmul",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_matmul,
            .adjoint = adjoint_matmul,
            .scalar = std::monostate{},
            .precision = Precision::lower
        };
    }

    MatmulShape matmul_shape(const Shape& lhs, const Shape& rhs) {
        if (lhs.rank() == 0 || rhs.rank() == 0) {
            ORMI_THROW(ShapeError, "Matmul requires tensors with at least one dimension.");
        }

        const bool lhs_vector = lhs.rank() == 1;
        const bool rhs_vector = rhs.rank() == 1;
        const auto lhs_inner = lhs[lhs.rank() - 1];
        const auto rhs_inner = rhs[rhs.rank() - (rhs_vector ? 1 : 2)];

        if (lhs_inner != rhs_inner) {
            ORMI_THROW(ShapeError, "Matmul inner dimensions must match.");
        }

        const auto& lhs_dims = lhs.dims();
        const auto& rhs_dims = rhs.dims();
        const auto lhs_matrix_rank = lhs_vector ? 1 : 2;
        const auto rhs_matrix_rank = rhs_vector ? 1 : 2;
        const Shape lhs_batch(std::vector<std::int64_t>(
            lhs_dims.begin(),
            lhs_dims.end() - lhs_matrix_rank
        ));
        const Shape rhs_batch(std::vector<std::int64_t>(
            rhs_dims.begin(),
            rhs_dims.end() - rhs_matrix_rank
        ));
        auto batch = broadcast(lhs_batch, rhs_batch);
        auto output_dims = batch.dims();
        const auto rows = static_cast<std::size_t>(lhs_vector ? 1 : lhs[lhs.rank() - 2]);
        const auto inner = static_cast<std::size_t>(lhs_inner);
        const auto columns = static_cast<std::size_t>(rhs_vector ? 1 : rhs[rhs.rank() - 1]);

        if (!lhs_vector) {
            output_dims.push_back(static_cast<std::int64_t>(rows));
        }

        if (!rhs_vector) {
            output_dims.push_back(static_cast<std::int64_t>(columns));
        }

        return MatmulShape{
            std::move(batch),
            Shape(std::move(output_dims)),
            rows,
            inner,
            columns,
            lhs_vector,
            rhs_vector
        };
    }

    std::shared_ptr<Tensor> matmul(
        const std::shared_ptr<Tensor>& lhs,
        const std::shared_ptr<Tensor>& rhs
    ) {
        if (!lhs || !rhs) {
            ORMI_THROW(OperationError, "Matmul requires two tensors.");
        }

        const auto lhs_device = lhs->device();

        if (lhs_device != rhs->device()) {
            ORMI_THROW(DeviceError, "Matmul requires tensors on the same device.");
        }

        const auto lhs_dtype = lhs->dtype();
        const auto rhs_dtype = rhs->dtype();

        if (!is_numeric(lhs_dtype) || !is_numeric(rhs_dtype)) {
            ORMI_THROW(DTypeError, "Matmul requires numeric tensors.");
        }

        auto shape = matmul_shape(lhs->shape(), rhs->shape());

        if (!precision::state.enabled && lhs_dtype == rhs_dtype) {
            return std::make_shared<Tensor>(
                op_ref<matmul_op>(),
                std::vector<std::shared_ptr<Tensor>>{lhs, rhs},
                lhs_device,
                std::move(shape.output),
                lhs_dtype
            );
        }

        const auto promoted = lhs_dtype == rhs_dtype
            ? lhs_dtype
            : promote(lhs_dtype, rhs_dtype);
        const auto dtype = precision::apply(matmul_op.precision, promoted);
        const auto promoted_lhs = lhs_dtype == dtype ? lhs : cast(lhs, dtype);
        const auto promoted_rhs = rhs_dtype == dtype ? rhs : cast(rhs, dtype);

        return std::make_shared<Tensor>(
            op_ref<matmul_op>(),
            std::vector<std::shared_ptr<Tensor>>{promoted_lhs, promoted_rhs},
            lhs_device,
            std::move(shape.output),
            dtype
        );
    }
}
