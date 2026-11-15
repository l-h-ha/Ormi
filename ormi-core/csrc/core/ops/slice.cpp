#include <cstddef>
#include <cstdint>
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
        std::size_t axis_index(std::int64_t axis, std::size_t rank) {
            if (axis < 0) {
                axis += static_cast<std::int64_t>(rank);
            }

            if (axis < 0 || axis >= static_cast<std::int64_t>(rank)) {
                ORMI_THROW(ShapeError, "Slice or concatenation axis is out of range.");
            }

            return static_cast<std::size_t>(axis);
        }

        std::size_t step_size(std::int64_t step) {
            return step < 0
                ? static_cast<std::size_t>(-(step + 1)) + 1
                : static_cast<std::size_t>(step);
        }

        std::shared_ptr<Tensor> make_slice(
            const std::shared_ptr<Tensor>& input, Slice attrs, std::int64_t length
        );

        // Copy contiguous trailing blocks, mapping the selected axis on one side.
        void copy_region(Tensor& out, const Tensor& input, const Slice& attrs, bool scatter) {
            if (input.numel() == 0 || out.numel() == 0) {
                return;
            }

            const auto axis = static_cast<std::size_t>(attrs.axis);
            auto shape = scatter ? input.shape().dims() : out.shape().dims();
            auto source_strides = std::vector<std::int64_t>(input.strides().begin(), input.strides().end());
            auto destination_strides = std::vector<std::int64_t>(out.strides().begin(), out.strides().end());

            if (attrs.squeeze) {
                shape.insert(shape.begin() + axis, 1);
                auto& strides = scatter ? source_strides : destination_strides;
                strides.insert(strides.begin() + axis, 0);
            }

            const auto elements = scatter ? input.numel() : out.numel();
            std::size_t block = 1;
            std::size_t dimensions = shape.size();

            while (dimensions > 0) {
                const auto dimension = dimensions - 1;

                if (shape[dimension] != 1) {
                    if ((dimension == axis && attrs.step != 1) ||
                        static_cast<std::size_t>(source_strides[dimension]) != block ||
                        static_cast<std::size_t>(destination_strides[dimension]) != block) {
                        break;
                    }
                }

                block *= static_cast<std::size_t>(shape[dimension]);
                --dimensions;
            }

            const auto item_size = dtype_size(out.dtype());
            auto* destination = static_cast<unsigned char*>(out.data());
            const auto* source = static_cast<const unsigned char*>(input.data());
            const auto& memory = runtime::backend(out.device()).memory;
            const auto step = step_size(attrs.step);

            for (std::size_t linear = 0; linear < elements; linear += block) {
                auto remaining = linear;
                std::size_t source_offset = 0;
                std::size_t destination_offset = 0;

                for (std::size_t dimension = shape.size(); dimension-- > 0;) {
                    const auto coordinate = remaining % static_cast<std::size_t>(shape[dimension]);
                    remaining /= static_cast<std::size_t>(shape[dimension]);
                    auto source_coordinate = coordinate;
                    auto destination_coordinate = coordinate;

                    if (dimension == axis) {
                        const auto selected = attrs.step > 0
                            ? static_cast<std::size_t>(attrs.start) + coordinate * step
                            : static_cast<std::size_t>(attrs.start) - coordinate * step;

                        if (scatter) {
                            destination_coordinate = selected;
                        } else {
                            source_coordinate = selected;
                        }
                    }

                    source_offset += source_coordinate * static_cast<std::size_t>(source_strides[dimension]);
                    destination_offset += destination_coordinate * static_cast<std::size_t>(destination_strides[dimension]);
                }

                memory.copy_async(out.device(), destination + destination_offset * item_size,
                                  source + source_offset * item_size, block * item_size);
            }
        }

        void compute_slice(Tensor& out) {
            copy_region(out, *out.parents()[0], std::get<Slice>(out.attrs()), false);
        }

        void compute_scatter(Tensor& out) {
            if (out.numel() == 0) {
                return;
            }

            const auto& memory = runtime::backend(out.device()).memory;
            memory.fill(out.device(), out.data(), 0, out.bytes());
            copy_region(out, *out.parents()[0], std::get<Slice>(out.attrs()), true);
        }

        void adjoint_slice(const std::shared_ptr<Tensor>& node, const std::shared_ptr<Tensor>& upstream, Grads& gradients);

        void adjoint_scatter(const std::shared_ptr<Tensor>& node, const std::shared_ptr<Tensor>& upstream, Grads& gradients) {
            const auto& parent = node->parents()[0];
            const auto& attrs = std::get<Slice>(node->attrs());
            const auto length = attrs.squeeze ? 1 : parent->shape()[static_cast<std::size_t>(attrs.axis)];
            gradients.add(parent, make_slice(upstream, attrs, length));
        }

        const Op slice_op{
            .name = "slice", .output = Output::alias, .parents = 1,
            .compute = nullptr, .adjoint = adjoint_slice, .scalar = std::monostate{}
        };
        const Op slice_copy_op{
            .name = "slice", .output = Output::allocate, .parents = 1,
            .compute = compute_slice, .adjoint = adjoint_slice, .scalar = std::monostate{}
        };
        const Op scatter_op{
            .name = "scatter_slice", .output = Output::allocate, .parents = 1,
            .compute = compute_scatter, .adjoint = adjoint_scatter, .scalar = std::monostate{}
        };

        void adjoint_slice(const std::shared_ptr<Tensor>& node, const std::shared_ptr<Tensor>& upstream, Grads& gradients) {
            const auto& parent = node->parents()[0];
            gradients.add(parent, std::make_shared<Tensor>(
                op_ref<scatter_op>(), node->attrs(), std::vector<std::shared_ptr<Tensor>>{upstream},
                parent->device(), parent->shape(), parent->dtype()
            ));
        }

        std::shared_ptr<Tensor> make_slice(const std::shared_ptr<Tensor>& input, Slice attrs, std::int64_t length) {
            if (!input) {
                ORMI_THROW(OperationError, "Slice requires a tensor.");
            }

            const auto axis = axis_index(attrs.axis, input->rank());
            attrs.axis = static_cast<std::int64_t>(axis);

            const auto extent = input->shape()[axis];

            if (length < 0 || attrs.step == 0) {
                ORMI_THROW(ShapeError, "Slice length must be non-negative and step must be nonzero.");
            }

            if (attrs.start < 0 || attrs.start > extent || (length > 0 && attrs.start == extent)) {
                ORMI_THROW(ShapeError, "Slice start is out of range.");
            }

            if (length > 0) {
                const auto distance = static_cast<std::size_t>(attrs.step > 0 ? extent - 1 - attrs.start : attrs.start);

                if (static_cast<std::size_t>(length - 1) > distance / step_size(attrs.step)) {
                    ORMI_THROW(ShapeError, "Slice range is out of bounds.");
                }
            }

            auto dimensions = input->shape().dims();
            dimensions[axis] = length;
            auto strides = std::vector<std::int64_t>(input->strides().begin(), input->strides().end());
            auto offset = input->offset_bytes();

            if (attrs.step > 0) {
                const auto item_size = dtype_size(input->dtype());
                const auto stride = static_cast<std::size_t>(strides[axis]);

                if (length != 0) {
                    const auto start = static_cast<std::size_t>(attrs.start);
                    const auto limit = (std::numeric_limits<std::size_t>::max() - offset) / item_size;

                    if (stride != 0 && start > limit / stride) {
                        ORMI_THROW(ShapeError, "Slice byte offset overflow.");
                    }

                    offset += start * stride * item_size;
                }

                if (strides[axis] > std::numeric_limits<std::int64_t>::max() / attrs.step) {
                    if (length > 1) {
                        ORMI_THROW(ShapeError, "Slice stride overflow.");
                    }

                    strides[axis] = 0;
                } else {
                    strides[axis] *= attrs.step;
                }
            }

            if (attrs.squeeze) {
                dimensions.erase(dimensions.begin() + axis);
                strides.erase(strides.begin() + axis);
            }

            if (attrs.step > 0) {
                return std::make_shared<Tensor>(op_ref<slice_op>(), attrs, input, Shape(std::move(dimensions)),
                                                std::move(strides), offset);
            }

            return std::make_shared<Tensor>(op_ref<slice_copy_op>(), attrs, std::vector<std::shared_ptr<Tensor>>{input},
                                            input->device(), Shape(std::move(dimensions)), input->dtype());
        }

        void compute_concat(Tensor& out) {
            const auto axis = std::get<Concat>(out.attrs()).axis;
            std::int64_t offset = 0;

            for (const auto& input : out.parents()) {
                copy_region(out, *input, Slice{axis, offset, 1}, true);
                offset += input->shape()[static_cast<std::size_t>(axis)];
            }
        }

        void adjoint_concat(const std::shared_ptr<Tensor>& node, const std::shared_ptr<Tensor>& upstream, Grads& gradients) {
            const auto axis = std::get<Concat>(node->attrs()).axis;
            std::int64_t offset = 0;

            for (const auto& parent : node->parents()) {
                const auto length = parent->shape()[static_cast<std::size_t>(axis)];
                gradients.add(parent, slice(upstream, axis, offset, length));
                offset += length;
            }
        }

        const Op concat_op{
            .name = "concat", .output = Output::allocate, .parents = variadic_parents,
            .compute = compute_concat, .adjoint = adjoint_concat, .scalar = std::monostate{}
        };
    }

    std::shared_ptr<Tensor> slice(const std::shared_ptr<Tensor>& input, std::int64_t axis,
                                  std::int64_t start, std::int64_t length, std::int64_t step) {
        return make_slice(input, Slice{axis, start, step}, length);
    }

    std::shared_ptr<Tensor> select_axis(const std::shared_ptr<Tensor>& input, std::int64_t axis, std::int64_t index) {
        if (!input) {
            ORMI_THROW(OperationError, "Axis selection requires a tensor.");
        }

        const auto selected_axis = axis_index(axis, input->rank());

        if (index < 0) {
            index += input->shape()[selected_axis];
        }

        return make_slice(input, Slice{static_cast<std::int64_t>(selected_axis), index, 1, true}, 1);
    }

    std::shared_ptr<Tensor> concat(const std::vector<std::shared_ptr<Tensor>>& inputs, std::int64_t axis) {
        if (inputs.empty()) {
            ORMI_THROW(OperationError, "Concatenation requires at least one tensor.");
        }

        for (const auto& input : inputs) {
            if (!input) {
                ORMI_THROW(OperationError, "Concatenation cannot contain a null tensor.");
            }
        }

        const auto& first = inputs.front();
        const auto selected_axis = axis_index(axis, first->rank());
        auto dimensions = first->shape().dims();
        dimensions[selected_axis] = 0;

        for (const auto& input : inputs) {
            if (input->device() != first->device()) {
                ORMI_THROW(DeviceError, "Concatenation tensors must use the same device.");
            }

            if (input->dtype() != first->dtype()) {
                ORMI_THROW(DTypeError, "Concatenation tensors must have the same dtype.");
            }

            if (input->rank() != first->rank()) {
                ORMI_THROW(ShapeError, "Concatenation tensors must have the same rank.");
            }

            for (std::size_t dimension = 0; dimension < input->rank(); ++dimension) {
                if (dimension != selected_axis && input->shape()[dimension] != dimensions[dimension]) {
                    ORMI_THROW(ShapeError, "Concatenation dimensions must match outside the selected axis.");
                }
            }

            const auto length = input->shape()[selected_axis];

            if (length > std::numeric_limits<std::int64_t>::max() - dimensions[selected_axis]) {
                ORMI_THROW(ShapeError, "Concatenation axis size overflow.");
            }

            dimensions[selected_axis] += length;
        }

        Shape shape(std::move(dimensions));

        if (shape.numel() > std::numeric_limits<std::size_t>::max() / dtype_size(first->dtype())) {
            ORMI_THROW(ShapeError, "Concatenation byte size overflow.");
        }

        return std::make_shared<Tensor>(op_ref<concat_op>(), Concat{static_cast<std::int64_t>(selected_axis)}, inputs,
                                        first->device(), std::move(shape), first->dtype());
    }
}
