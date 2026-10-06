#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/attrs.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    namespace {
        void adjoint_reshape(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], reshape(upstream, node->parents()[0]->shape()));
        }

        void adjoint_transpose(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& axes = std::get<Permute>(node->attrs()).axes;
            std::vector<std::int64_t> inverse(axes.size());

            for (std::size_t axis = 0; axis < axes.size(); ++axis) {
                inverse[static_cast<std::size_t>(axes[axis])] = static_cast<std::int64_t>(axis);
            }

            gradients.add(node->parents()[0], transpose(upstream, std::move(inverse)));
        }

        void adjoint_expand(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], unbroadcast(upstream, node->parents()[0]->shape()));
        }

        const Op reshape_op{
            .name = "reshape",
            .output = Output::alias,
            .parents = 1,
            .compute = nullptr,
            .adjoint = adjoint_reshape,
            .scalar = std::monostate{}
        };
        const Op transpose_op{
            .name = "transpose",
            .output = Output::alias,
            .parents = 1,
            .compute = nullptr,
            .adjoint = adjoint_transpose,
            .scalar = std::monostate{}
        };
        const Op expand_op{
            .name = "expand",
            .output = Output::alias,
            .parents = 1,
            .compute = nullptr,
            .adjoint = adjoint_expand,
            .scalar = std::monostate{}
        };

        std::vector<std::int64_t> normalize_axes(
            std::vector<std::int64_t> axes,
            std::size_t rank
        ) {
            if (axes.size() != rank) {
                ORMI_THROW(ShapeError, "Transpose requires one axis for each input dimension.");
            }

            for (auto& axis : axes) {
                if (axis < 0) {
                    axis += static_cast<std::int64_t>(rank);
                }

                if (axis < 0 || axis >= static_cast<std::int64_t>(rank)) {
                    ORMI_THROW(ShapeError, "Transpose axis is out of range.");
                }
            }

            auto sorted = axes;
            std::sort(sorted.begin(), sorted.end());

            for (std::size_t axis = 0; axis < rank; ++axis) {
                if (sorted[axis] != static_cast<std::int64_t>(axis)) {
                    ORMI_THROW(ShapeError, "Transpose axes must be a permutation.");
                }
            }

            return axes;
        }

        std::optional<std::vector<std::int64_t>> reshape_strides(
            const Tensor& input,
            const Shape& shape
        ) {
            if (input.numel() == 0 || input.rank() == 0) {
                return contiguous_strides(shape);
            }

            std::vector<std::int64_t> strides(shape.rank());
            const auto input_strides = input.strides();
            auto output_axis = shape.rank();
            std::int64_t chunk_stride = input_strides.back();
            std::int64_t input_elements = 1;
            std::int64_t output_elements = 1;

            for (std::size_t axis = input.rank(); axis > 0; --axis) {
                const auto input_axis = axis - 1;
                input_elements *= input.shape()[input_axis];
                const bool chunk_ends =
                    input_axis == 0 ||
                    (
                        input.shape()[input_axis - 1] != 1 &&
                        input_strides[input_axis - 1] != input_elements * chunk_stride
                    );

                if (!chunk_ends) {
                    continue;
                }

                while (
                    output_axis > 0 &&
                    (
                        output_elements < input_elements ||
                        shape[output_axis - 1] == 1
                    )
                ) {
                    --output_axis;
                    strides[output_axis] = output_elements * chunk_stride;
                    output_elements *= shape[output_axis];
                }

                if (output_elements != input_elements) {
                    return std::nullopt;
                }

                if (input_axis > 0) {
                    chunk_stride = input_strides[input_axis - 1];
                    input_elements = 1;
                    output_elements = 1;
                }
            }

            if (output_axis != 0) {
                return std::nullopt;
            }

            return strides;
        }
    }

    std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& input, Shape shape) {
        if (!input) {
            ORMI_THROW(OperationError, "Reshape requires a tensor.");
        }

        if (input->numel() != shape.numel()) {
            ORMI_THROW(ShapeError, "Reshape must preserve the number of elements.");
        }

        auto strides = reshape_strides(*input, shape);
        if (!strides) {
            ORMI_THROW(ShapeError, "Reshape is incompatible with the tensor's storage layout.");
        }

        return std::make_shared<Tensor>(
            op_ref<reshape_op>(),
            std::monostate{},
            input,
            std::move(shape),
            std::move(*strides),
            input->offset_bytes()
        );
    }

    std::shared_ptr<Tensor> transpose(
        const std::shared_ptr<Tensor>& input,
        std::vector<std::int64_t> axes
    ) {
        if (!input) {
            ORMI_THROW(OperationError, "Transpose requires a tensor.");
        }

        axes = normalize_axes(std::move(axes), input->rank());
        std::vector<std::int64_t> dims(input->rank());
        std::vector<std::int64_t> strides(input->rank());
        const auto input_strides = input->strides();

        for (std::size_t axis = 0; axis < input->rank(); ++axis) {
            const auto input_axis = static_cast<std::size_t>(axes[axis]);
            dims[axis] = input->shape()[input_axis];
            strides[axis] = input_strides[input_axis];
        }

        return std::make_shared<Tensor>(
            op_ref<transpose_op>(),
            Permute{std::move(axes)},
            input,
            Shape(std::move(dims)),
            std::move(strides),
            input->offset_bytes()
        );
    }

    std::shared_ptr<Tensor> expand(const std::shared_ptr<Tensor>& input, Shape shape) {
        if (!input) {
            ORMI_THROW(OperationError, "Expand requires a tensor.");
        }

        if (shape.rank() < input->rank()) {
            ORMI_THROW(ShapeError, "Expand cannot remove dimensions.");
        }

        const auto rank_offset = shape.rank() - input->rank();
        const auto input_strides = input->strides();
        std::vector<std::int64_t> strides(shape.rank(), 0);

        for (std::size_t axis = rank_offset; axis < shape.rank(); ++axis) {
            const auto input_axis = axis - rank_offset;
            const auto input_dim = input->shape()[input_axis];
            const auto output_dim = shape[axis];

            if (input_dim == output_dim) {
                strides[axis] = input_strides[input_axis];
            } else if (input_dim != 1) {
                ORMI_THROW(ShapeError, "Expand dimensions must match or expand an input dimension of one.");
            }
        }

        return std::make_shared<Tensor>(
            op_ref<expand_op>(),
            std::monostate{},
            input,
            std::move(shape),
            std::move(strides),
            input->offset_bytes()
        );
    }
}
