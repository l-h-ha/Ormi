#include <memory>
#include <utility>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/pointwise.h"

namespace ormi::core::ops {
    namespace {
        void compute_binary(BinaryOp op, Tensor& out) {
            runtime::pointwise::binary(op, out, *out.parents()[0], *out.parents()[1]);
        }

        void compute_add(Tensor& out) {
            compute_binary(BinaryOp::add, out);
        }

        void compute_sub(Tensor& out) {
            compute_binary(BinaryOp::sub, out);
        }

        void compute_mul(Tensor& out) {
            compute_binary(BinaryOp::mul, out);
        }

        void compute_div(Tensor& out) {
            compute_binary(BinaryOp::div, out);
        }

        void adjoint_add(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& lhs = node->parents()[0];
            const auto& rhs = node->parents()[1];
            gradients.add(lhs, unbroadcast(upstream, lhs->shape()));
            gradients.add(rhs, unbroadcast(upstream, rhs->shape()));
        }

        void adjoint_sub(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& lhs = node->parents()[0];
            const auto& rhs = node->parents()[1];
            gradients.add(lhs, unbroadcast(upstream, lhs->shape()));
            gradients.add(rhs, unbroadcast(neg(upstream), rhs->shape()));
        }

        void adjoint_mul(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& lhs = node->parents()[0];
            const auto& rhs = node->parents()[1];
            const auto lhs_gradient = mul(upstream, rhs);
            const auto rhs_gradient = mul(upstream, lhs);
            gradients.add(lhs, unbroadcast(lhs_gradient, lhs->shape()));
            gradients.add(rhs, unbroadcast(rhs_gradient, rhs->shape()));
        }

        void adjoint_div(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& lhs = node->parents()[0];
            const auto& rhs = node->parents()[1];
            const auto lhs_gradient = div(upstream, rhs);
            const auto rhs_squared = mul(rhs, rhs);
            const auto rhs_gradient = neg(div(mul(upstream, lhs), rhs_squared));
            gradients.add(lhs, unbroadcast(lhs_gradient, lhs->shape()));
            gradients.add(rhs, unbroadcast(rhs_gradient, rhs->shape()));
        }

        void adjoint_minimum(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& lhs = node->parents()[0];
            const auto& rhs = node->parents()[1];
            const auto direction = sign(sub(lhs, rhs));
            const auto two = add(ones_like(direction), ones_like(direction));
            const auto left = div(sub(ones_like(direction), direction), two);
            gradients.add(lhs, unbroadcast(mul(upstream, left), lhs->shape()));
            gradients.add(rhs, unbroadcast(mul(upstream, sub(ones_like(left), left)), rhs->shape()));
        }

        void compute_minimum(Tensor& out) {
            compute_binary(BinaryOp::minimum, out);
        }

        const Op minimum_op{
            .name = "minimum",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_minimum,
            .adjoint = adjoint_minimum,
            .scalar = BinaryOp::minimum
        };

        void adjoint_maximum(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& lhs = node->parents()[0];
            const auto& rhs = node->parents()[1];
            const auto direction = sign(sub(lhs, rhs));
            const auto two = add(ones_like(direction), ones_like(direction));
            const auto left = div(add(ones_like(direction), direction), two);
            gradients.add(lhs, unbroadcast(mul(upstream, left), lhs->shape()));
            gradients.add(rhs, unbroadcast(mul(upstream, sub(ones_like(left), left)), rhs->shape()));
        }

        void compute_maximum(Tensor& out) {
            compute_binary(BinaryOp::maximum, out);
        }

        const Op maximum_op{
            .name = "maximum",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_maximum,
            .adjoint = adjoint_maximum,
            .scalar = BinaryOp::maximum
        };

        const Op add_op{
            .name = "add",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_add,
            .adjoint = adjoint_add,
            .scalar = BinaryOp::add
        };
        const Op sub_op{
            .name = "sub",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_sub,
            .adjoint = adjoint_sub,
            .scalar = BinaryOp::sub
        };
        const Op mul_op{
            .name = "mul",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_mul,
            .adjoint = adjoint_mul,
            .scalar = BinaryOp::mul
        };
        const Op div_op{
            .name = "div",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_div,
            .adjoint = adjoint_div,
            .scalar = BinaryOp::div
        };

        std::shared_ptr<Tensor> binary(
            OpRef op,
            const std::shared_ptr<Tensor>& lhs,
            const std::shared_ptr<Tensor>& rhs
        ) {
            if (!lhs || !rhs) {
                ORMI_THROW(OperationError, "Pointwise binary operations require two tensors.");
            }

            const auto lhs_device = lhs->device();

            if (lhs_device != rhs->device()) {
                ORMI_THROW(DeviceError, "Pointwise binary operations require tensors on the same device.");
            }

            const auto lhs_dtype = lhs->dtype();
            const auto rhs_dtype = rhs->dtype();

            if (lhs_dtype != rhs_dtype) {
                if (!is_numeric(lhs_dtype) || !is_numeric(rhs_dtype)) {
                    ORMI_THROW(DTypeError, "Pointwise binary operations require numeric tensors.");
                }

                const auto dtype = promote(lhs_dtype, rhs_dtype);
                const auto shape = broadcast(lhs->shape(), rhs->shape());

                if (op.get() == &mul_op) {
                    if (is_one(*lhs) && rhs->shape() == shape) {
                        return rhs_dtype == dtype ? rhs : cast(rhs, dtype);
                    }

                    if (is_one(*rhs) && lhs->shape() == shape) {
                        return lhs_dtype == dtype ? lhs : cast(lhs, dtype);
                    }
                }

                if (op.get() == &div_op && is_one(*rhs) && lhs->shape() == shape) {
                    return lhs_dtype == dtype ? lhs : cast(lhs, dtype);
                }

                const auto promoted_lhs = lhs_dtype == dtype ? lhs : cast(lhs, dtype);
                const auto promoted_rhs = rhs_dtype == dtype ? rhs : cast(rhs, dtype);
                return binary(op, promoted_lhs, promoted_rhs);
            }

            if (!is_numeric(lhs_dtype)) {
                ORMI_THROW(DTypeError, "Pointwise binary operations require numeric tensors.");
            }

            auto shape = broadcast(lhs->shape(), rhs->shape());

            if (op.get() == &mul_op) {
                if (is_one(*lhs) && rhs->shape() == shape) {
                    return rhs;
                }

                if (is_one(*rhs) && lhs->shape() == shape) {
                    return lhs;
                }
            }

            if (op.get() == &div_op && is_one(*rhs) && lhs->shape() == shape) {
                return lhs;
            }

            return std::make_shared<Tensor>(
                op,
                std::vector<std::shared_ptr<Tensor>>{lhs, rhs},
                lhs_device,
                std::move(shape),
                lhs_dtype
            );
        }
    }

    std::shared_ptr<Tensor> minimum(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) {
        if ((lhs && !is_floating(lhs->dtype())) || (rhs && !is_floating(rhs->dtype()))) {
            ORMI_THROW(DTypeError, "Minimum requires floating-point tensors.");
        }
        return binary(op_ref<minimum_op>(), lhs, rhs);
    }

    std::shared_ptr<Tensor> maximum(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) {
        if ((lhs && !is_floating(lhs->dtype())) || (rhs && !is_floating(rhs->dtype()))) {
            ORMI_THROW(DTypeError, "Maximum requires floating-point tensors.");
        }
        return binary(op_ref<maximum_op>(), lhs, rhs);
    }

    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) {
        return binary(op_ref<add_op>(), lhs, rhs);
    }

    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) {
        return binary(op_ref<sub_op>(), lhs, rhs);
    }

    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) {
        return binary(op_ref<mul_op>(), lhs, rhs);
    }

    std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) {
        return binary(op_ref<div_op>(), lhs, rhs);
    }
}
