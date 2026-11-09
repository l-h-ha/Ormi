#include <memory>
#include <utility>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/index.h>

#include "core/runtime/device.h"

namespace ormi::core::ops {
    namespace {
        void compute_pick(Tensor& out) {
            runtime::backend(out.device())
                .ops.pick(out, *out.parents()[0], *out.parents()[1], std::get<Pick>(out.attrs()));
        }

        void compute_unpick(Tensor& out) {
            runtime::backend(out.device())
                .ops.unpick(out, *out.parents()[0], *out.parents()[1], std::get<Pick>(out.attrs()));
        }

        void adjoint_pick(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        );

        void adjoint_unpick(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& attrs = std::get<Pick>(node->attrs());
            gradients.add(
                node->parents()[0],
                pick(upstream, node->parents()[1], attrs.axis, attrs.ignore_index)
            );
        }

        const Op pick_op{
            .name = "pick",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_pick,
            .adjoint = adjoint_pick,
            .scalar = std::monostate{}
        };
        const Op unpick_op{
            .name = "unpick",
            .output = Output::allocate,
            .parents = 2,
            .compute = compute_unpick,
            .adjoint = adjoint_unpick,
            .scalar = std::monostate{}
        };

        void adjoint_pick(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& attrs = std::get<Pick>(node->attrs());
            const auto gradient = unpick(
                upstream, node->parents()[1], attrs.shape, attrs.axis, attrs.ignore_index
            );
            gradients.add(node->parents()[0], gradient);
        }
    }

    std::shared_ptr<Tensor> pick(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& indices,
        std::int64_t axis,
        std::optional<std::int32_t> ignore_index
    ) {
        if (!input || !indices) {
            ORMI_THROW(OperationError, "Pick requires input and indices.");
        }
        if (!is_floating(input->dtype()) || indices->dtype() != DType::int32) {
            ORMI_THROW(DTypeError, "Pick requires floating-point input and Int32 indices.");
        }
        if (input->device() != indices->device()) {
            ORMI_THROW(DeviceError, "Pick tensors must use the same device.");
        }
        if (axis < 0) {
            axis += static_cast<std::int64_t>(input->rank());
        }
        if (axis < 0 || axis >= static_cast<std::int64_t>(input->rank())) {
            ORMI_THROW(ShapeError, "Pick axis is out of range.");
        }
        auto shape = input->shape().dims();
        shape.erase(shape.begin() + axis);
        if (Shape(shape) != indices->shape()) {
            ORMI_THROW(ShapeError, "Pick indices must match the input shape with the selected axis removed.");
        }
        return std::make_shared<Tensor>(
            op_ref<pick_op>(),
            Pick{axis, input->shape(), ignore_index},
            std::vector<std::shared_ptr<Tensor>>{input, indices},
            input->device(),
            indices->shape(),
            input->dtype()
        );
    }

    std::shared_ptr<Tensor> unpick(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& indices,
        Shape shape,
        std::int64_t axis,
        std::optional<std::int32_t> ignore_index
    ) {
        if (!input || !indices) {
            ORMI_THROW(OperationError, "Unpick requires input and indices.");
        }
        if (!is_floating(input->dtype()) || indices->dtype() != DType::int32) {
            ORMI_THROW(DTypeError, "Unpick requires floating-point input and Int32 indices.");
        }
        if (input->device() != indices->device()) {
            ORMI_THROW(DeviceError, "Unpick tensors must use the same device.");
        }
        if (axis < 0) {
            axis += static_cast<std::int64_t>(shape.rank());
        }
        if (axis < 0 || axis >= static_cast<std::int64_t>(shape.rank())) {
            ORMI_THROW(ShapeError, "Unpick axis is out of range.");
        }
        auto dimensions = shape.dims();
        dimensions.erase(dimensions.begin() + axis);
        if (Shape(dimensions) != input->shape() || input->shape() != indices->shape()) {
            ORMI_THROW(ShapeError, "Unpick input and indices must match the output shape with the selected axis removed.");
        }
        return std::make_shared<Tensor>(
            op_ref<unpick_op>(),
            Pick{axis, shape, ignore_index},
            std::vector<std::shared_ptr<Tensor>>{input, indices},
            input->device(),
            shape,
            input->dtype()
        );
    }
}
