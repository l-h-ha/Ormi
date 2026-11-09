#include <memory>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>

#include "core/precision.h"
#include "core/runtime/dispatch/pointwise.h"

namespace ormi::core::ops {
    namespace {
        std::shared_ptr<Tensor> scalar(const std::shared_ptr<Tensor>& input, double value) {
            Scalar scalar_value = value;
            if (input->dtype() == DType::float32) {
                scalar_value = static_cast<float>(value);
            }
            if (input->dtype() == DType::float16) {
                scalar_value = Float16(static_cast<float>(value));
            }
            return full(Shape{}, scalar_value, input->device());
        }

        template <UnaryOp op>
        void compute(Tensor& out) {
            runtime::pointwise::unary(op, out, *out.parents()[0]);
        }

        void adjoint_abs(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            gradients.add(input, mul(upstream, sign(input)));
        }

        // Skip the inputs of discrete masks because their derivative is zero.
        void adjoint_discrete(const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&, Grads&) {}

        void adjoint_relu(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            gradients.add(input, mul(upstream, positive(input)));
        }

        void adjoint_tanh(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            const auto one = ones_like(node);
            const auto squared_output = mul(node, node);
            const auto derivative = sub(one, squared_output);

            gradients.add(input, mul(upstream, derivative));
        }

        void adjoint_sigmoid(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            const auto complement = sub(ones_like(node), node);
            const auto derivative = mul(node, complement);

            gradients.add(input, mul(upstream, derivative));
        }

        void adjoint_softplus(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            gradients.add(input, mul(upstream, sigmoid(input)));
        }

        void adjoint_erf(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            constexpr double two_over_sqrt_pi = 1.1283791670955125739;
            const auto& input = node->parents()[0];
            const auto factor = scalar(input, two_over_sqrt_pi);
            const auto squared_input = mul(input, input);
            const auto decay = exp(neg(squared_input));
            const auto derivative = mul(factor, decay);

            gradients.add(input, mul(upstream, derivative));
        }

        void adjoint_log1p(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            const auto denominator = add(ones_like(input), input);

            gradients.add(input, div(upstream, denominator));
        }

        void adjoint_expm1(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            gradients.add(input, mul(upstream, exp(input)));
        }

        void adjoint_rsqrt(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            const auto squared_output = mul(node, node);
            const auto cubed_output = mul(squared_output, node);
            const auto derivative = mul(scalar(input, -0.5), cubed_output);

            gradients.add(input, mul(upstream, derivative));
        }

        const Op abs_op{
            .name = "abs",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::abs>,
            .adjoint = adjoint_abs,
            .scalar = UnaryOp::abs,
            .precision = Precision::float32
        };

        const Op sign_op{
            .name = "sign",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::sign>,
            .adjoint = adjoint_discrete,
            .scalar = UnaryOp::sign,
            .precision = Precision::float32
        };

        const Op positive_op{
            .name = "positive",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::positive>,
            .adjoint = adjoint_discrete,
            .scalar = UnaryOp::positive,
            .precision = Precision::float32
        };

        const Op relu_op{
            .name = "relu",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::relu>,
            .adjoint = adjoint_relu,
            .scalar = UnaryOp::relu,
            .precision = Precision::float32
        };

        const Op tanh_op{
            .name = "tanh",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::tanh>,
            .adjoint = adjoint_tanh,
            .scalar = UnaryOp::tanh,
            .precision = Precision::float32
        };

        const Op sigmoid_op{
            .name = "sigmoid",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::sigmoid>,
            .adjoint = adjoint_sigmoid,
            .scalar = UnaryOp::sigmoid,
            .precision = Precision::float32
        };

        const Op softplus_op{
            .name = "softplus",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::softplus>,
            .adjoint = adjoint_softplus,
            .scalar = UnaryOp::softplus,
            .precision = Precision::float32
        };

        const Op erf_op{
            .name = "erf",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::erf>,
            .adjoint = adjoint_erf,
            .scalar = UnaryOp::erf,
            .precision = Precision::float32
        };

        const Op log1p_op{
            .name = "log1p",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::log1p>,
            .adjoint = adjoint_log1p,
            .scalar = UnaryOp::log1p,
            .precision = Precision::float32
        };

        const Op expm1_op{
            .name = "expm1",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::expm1>,
            .adjoint = adjoint_expm1,
            .scalar = UnaryOp::expm1,
            .precision = Precision::float32
        };

        const Op rsqrt_op{
            .name = "rsqrt",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute<UnaryOp::rsqrt>,
            .adjoint = adjoint_rsqrt,
            .scalar = UnaryOp::rsqrt,
            .precision = Precision::float32
        };

        std::shared_ptr<Tensor> make_unary(OpRef op, const std::shared_ptr<Tensor>& input) {
            if (!input) {
                ORMI_THROW(OperationError, "Unary operations require a tensor.");
            }
            if (!is_floating(input->dtype())) {
                ORMI_THROW(DTypeError, "This unary operation requires a floating-point tensor.");
            }
            auto dtype = input->dtype();
            if (precision::state.enabled) {
                dtype = precision::apply(op.get()->precision, dtype);
            }

            const auto parent = dtype == input->dtype() ? input : cast(input, dtype);
            return std::make_shared<Tensor>(
                op,
                std::vector<std::shared_ptr<Tensor>>{parent},
                input->device(),
                input->shape(),
                dtype
            );
        }
    }

    std::shared_ptr<Tensor> abs(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<abs_op>(), input);
    }

    std::shared_ptr<Tensor> sign(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<sign_op>(), input);
    }

    std::shared_ptr<Tensor> positive(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<positive_op>(), input);
    }

    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<relu_op>(), input);
    }

    std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<tanh_op>(), input);
    }

    std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<sigmoid_op>(), input);
    }

    std::shared_ptr<Tensor> softplus(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<softplus_op>(), input);
    }

    std::shared_ptr<Tensor> erf(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<erf_op>(), input);
    }

    std::shared_ptr<Tensor> log1p(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<log1p_op>(), input);
    }

    std::shared_ptr<Tensor> expm1(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<expm1_op>(), input);
    }

    std::shared_ptr<Tensor> rsqrt(const std::shared_ptr<Tensor>& input) {
        return make_unary(op_ref<rsqrt_op>(), input);
    }

}
