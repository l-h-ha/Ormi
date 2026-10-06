#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    namespace {
        void adjoint_constant(
            const std::shared_ptr<Tensor>&,
            const std::shared_ptr<Tensor>&,
            Grads&
        ) {}

        DType scalar_dtype(const Scalar& value) {
            if (std::holds_alternative<double>(value)) {
                return DType::float64;
            }
            if (std::holds_alternative<float>(value)) {
                return DType::float32;
            }
            if (std::holds_alternative<Float16>(value)) {
                return DType::float16;
            }
            if (std::holds_alternative<std::int32_t>(value)) {
                return DType::int32;
            }
            return DType::boolean;
        }

        const void* scalar_data(const Scalar& value) {
            if (const auto* scalar = std::get_if<double>(&value)) {
                return scalar;
            }
            if (const auto* scalar = std::get_if<float>(&value)) {
                return scalar;
            }
            if (const auto* scalar = std::get_if<Float16>(&value)) {
                return scalar;
            }
            if (const auto* scalar = std::get_if<std::int32_t>(&value)) {
                return scalar;
            }
            return std::get_if<bool>(&value);
        }

        void compute_full(Tensor& out) {
            const auto& value = std::get<Scalar>(out.attrs());

            memory::fill_scalar(
                out.data(),
                out.device(),
                scalar_data(value),
                out.numel(),
                out.dtype()
            );
        }

        void compute_zeros(Tensor& out) {
            memory::zero(*out.buffer());
        }

        void compute_ones(Tensor& out) {
            switch (out.dtype()) {
                case DType::float64: {
                    constexpr double one = 1.0;
                    memory::fill_scalar(out.data(), out.device(), &one, out.numel(), out.dtype());
                    return;
                }
                case DType::float32: {
                    constexpr float one = 1.0f;
                    memory::fill_scalar(out.data(), out.device(), &one, out.numel(), out.dtype());
                    return;
                }
                case DType::float16: {
                    constexpr auto one = Float16::from_bits(0x3c00u);
                    memory::fill_scalar(out.data(), out.device(), &one, out.numel(), out.dtype());
                    return;
                }
                case DType::int32: {
                    constexpr std::int32_t one = 1;
                    memory::fill_scalar(out.data(), out.device(), &one, out.numel(), out.dtype());
                    return;
                }
                case DType::boolean: {
                    constexpr bool one = true;
                    memory::fill_scalar(out.data(), out.device(), &one, out.numel(), out.dtype());
                    return;
                }
            }

            ORMI_THROW(DTypeError, "Cannot create ones for an unknown dtype.");
        }

        const Op full_op{
            .name = "full",
            .output = Output::allocate,
            .parents = 0,
            .compute = compute_full,
            .adjoint = adjoint_constant,
            .scalar = std::monostate{}
        };
        const Op zeros_op{
            .name = "zeros_like",
            .output = Output::allocate,
            .parents = 0,
            .compute = compute_zeros,
            .adjoint = adjoint_constant,
            .scalar = std::monostate{}
        };
        const Op ones_op{
            .name = "ones_like",
            .output = Output::allocate,
            .parents = 0,
            .compute = compute_ones,
            .adjoint = adjoint_constant,
            .scalar = std::monostate{}
        };

        std::shared_ptr<Tensor> like(OpRef op, const std::shared_ptr<Tensor>& input) {
            if (!input) {
                ORMI_THROW(OperationError, "Tensor creation requires an input tensor.");
            }

            return std::make_shared<Tensor>(
                op,
                std::vector<std::shared_ptr<Tensor>>{},
                input->device(),
                input->shape(),
                input->dtype()
            );
        }
    }

    std::shared_ptr<Tensor> full(Shape shape, Scalar value, Device device) {
        const auto dtype = scalar_dtype(value);

        return std::make_shared<Tensor>(
            op_ref<full_op>(),
            Attrs{std::move(value)},
            std::vector<std::shared_ptr<Tensor>>{},
            device,
            std::move(shape),
            dtype
        );
    }

    std::shared_ptr<Tensor> zeros_like(const std::shared_ptr<Tensor>& input) {
        return like(op_ref<zeros_op>(), input);
    }

    std::shared_ptr<Tensor> ones_like(const std::shared_ptr<Tensor>& input) {
        return like(op_ref<ones_op>(), input);
    }

    bool is_zero(const Tensor& tensor) {
        return tensor.op() == &zeros_op;
    }

    bool is_one(const Tensor& tensor) {
        return tensor.op() == &ones_op;
    }
}
