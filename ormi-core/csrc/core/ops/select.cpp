#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/index.h>

#include "core/runtime/device.h"

namespace ormi::core::ops {
    namespace {
        std::shared_ptr<Tensor> scatter(const std::shared_ptr<Tensor>& scalar, Shape shape, Select attrs);

        void compute_scatter(Tensor& out) {
            const auto index = std::get<Select>(out.attrs()).index;
            const auto& scalar = *out.parents()[0];
            const auto& backend = runtime::backend(out.device());
            if (backend.ops.scatter_scalar) {
                backend.ops.scatter_scalar(out, scalar, index);
                return;
            }

            const auto item_size = dtype_size(out.dtype());
            auto* data = static_cast<unsigned char*>(out.data());
            backend.memory.fill(out.device(), data, 0, out.bytes());
            backend.memory.copy_async(out.device(), data + index * item_size, scalar.data(), item_size);
        }

        void adjoint_select(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& parent = node->parents()[0];
            gradients.add(parent, scatter(upstream, parent->shape(), std::get<Select>(node->attrs())));
        }

        void adjoint_scatter(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], select(upstream, std::get<Select>(node->attrs()).index));
        }

        const Op select_op{
            .name = "select",
            .output = Output::alias,
            .parents = 1,
            .compute = nullptr,
            .adjoint = adjoint_select,
            .scalar = std::monostate{}
        };
        const Op scatter_op{
            .name = "scatter_scalar",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_scatter,
            .adjoint = adjoint_scatter,
            .scalar = std::monostate{}
        };

        std::shared_ptr<Tensor> scatter(const std::shared_ptr<Tensor>& scalar, Shape shape, Select attrs) {
            return std::make_shared<Tensor>(
                op_ref<scatter_op>(),
                attrs,
                std::vector<std::shared_ptr<Tensor>>{scalar},
                scalar->device(),
                std::move(shape),
                scalar->dtype()
            );
        }
    }

    std::shared_ptr<Tensor> select(const std::shared_ptr<Tensor>& input, std::size_t index) {
        if (!input) {
            ORMI_THROW(OperationError, "Scalar selection requires a tensor.");
        }
        if (index >= input->numel()) {
            ORMI_THROW(ShapeError, "Scalar selection index is out of range.");
        }

        const auto item_size = dtype_size(input->dtype());
        auto offset = input->offset_bytes();
        auto remaining = index;
        const auto strides = input->strides();
        for (std::size_t axis = input->rank(); axis > 0; --axis) {
            const auto size = static_cast<std::size_t>(input->shape()[axis - 1]);
            const auto coordinate = remaining % size;
            remaining /= size;
            const auto stride = static_cast<std::size_t>(strides[axis - 1]);
            const auto limit = (std::numeric_limits<std::size_t>::max() - offset) / item_size;
            if (stride != 0 && coordinate > limit / stride) {
                ORMI_THROW(ShapeError, "Scalar selection byte offset overflow.");
            }
            offset += coordinate * stride * item_size;
        }
        if (offset > std::numeric_limits<std::size_t>::max() - item_size) {
            ORMI_THROW(ShapeError, "Scalar selection byte offset overflow.");
        }

        return std::make_shared<Tensor>(
            op_ref<select_op>(),
            Select{index},
            input,
            Shape{},
            std::vector<std::int64_t>{},
            offset
        );
    }
}
