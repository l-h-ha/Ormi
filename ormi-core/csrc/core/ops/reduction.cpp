#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/attrs.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/reduction.h"
#include "core/runtime/dispatch/finite.h"
#include "core/runtime/dispatch/memory.h"
#include "core/precision.h"

namespace ormi::core::ops {
    namespace {
        void compute_all_finite(Tensor& out) {
            const std::array<const Tensor*, 1> inputs{out.parents()[0].get()};
            const bool finite = runtime::finite::all(inputs);
            runtime::memory::fill_scalar(
                out.device(),
                out.data(),
                &finite,
                1,
                DType::boolean
            );
        }

        void compute_sum(Tensor& out) {
            const auto& attrs = std::get<Reduce>(out.attrs());
            runtime::reduction::reduce(
                ReduceOp::sum,
                out,
                *out.parents()[0],
                attrs.axes,
                attrs.keepdims
            );
        }

        void compute_mean(Tensor& out) {
            const auto& attrs = std::get<Reduce>(out.attrs());
            runtime::reduction::reduce(
                ReduceOp::mean,
                out,
                *out.parents()[0],
                attrs.axes,
                attrs.keepdims
            );
        }

        std::shared_ptr<Tensor> expanded_adjoint(
            const std::shared_ptr<Tensor>& node,
            std::shared_ptr<Tensor> gradient
        ) {
            const auto& input = node->parents()[0];
            const auto& attrs = std::get<Reduce>(node->attrs());

            if (!attrs.keepdims) {
                auto dims = input->shape().dims();

                for (const auto axis : attrs.axes) {
                    dims[static_cast<std::size_t>(axis)] = 1;
                }

                gradient = reshape(gradient, Shape(std::move(dims)));
            }

            return expand(gradient, input->shape());
        }

        void adjoint_sum(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            gradients.add(node->parents()[0], expanded_adjoint(node, upstream));
        }

        Scalar reciprocal_scalar(std::size_t count, DType dtype) {
            switch (dtype) {
                case DType::float64:
                    return 1.0 / static_cast<double>(count);
                case DType::float32:
                    return 1.0f / static_cast<float>(count);
                case DType::float16:
                    return Float16{1.0f / static_cast<float>(count)};
                case DType::int32:
                case DType::boolean:
                    ORMI_THROW(DTypeError, "Mean requires a floating-point tensor.");
            }

            ORMI_THROW(DTypeError, "Mean received an unknown dtype.");
        }

        void adjoint_mean(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            const auto& attrs = std::get<Reduce>(node->attrs());
            std::size_t count = 1;

            for (const auto axis : attrs.axes) {
                count *= static_cast<std::size_t>(
                    input->shape()[static_cast<std::size_t>(axis)]
                );
            }

            const auto scaled = count == 1 ? upstream : mul(
                upstream,
                full(
                    Shape{},
                    reciprocal_scalar(count, upstream->dtype()),
                    upstream->device()
                )
            );
            gradients.add(
                input,
                expanded_adjoint(node, scaled)
            );
        }

        const Op all_finite_op{
            .name = "all_finite",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_all_finite,
            .adjoint = nullptr,
            .scalar = std::monostate{}
        };
        const Op sum_op{
            .name = "sum",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_sum,
            .adjoint = adjoint_sum,
            .scalar = std::monostate{},
            .precision = Precision::float32
        };
        const Op mean_op{
            .name = "mean",
            .output = Output::allocate,
            .parents = 1,
            .compute = compute_mean,
            .adjoint = adjoint_mean,
            .scalar = std::monostate{},
            .precision = Precision::float32
        };

        std::vector<std::int64_t> normalize_axes(
            std::vector<std::int64_t> axes,
            std::size_t rank
        ) {
            if (axes.empty()) {
                axes.resize(rank);
                std::iota(axes.begin(), axes.end(), std::int64_t{0});
                return axes;
            }

            for (auto& axis : axes) {
                if (axis < 0) {
                    axis += static_cast<std::int64_t>(rank);
                }

                if (axis < 0 || axis >= static_cast<std::int64_t>(rank)) {
                    ORMI_THROW(ShapeError, "Reduction axis is out of range.");
                }
            }

            std::sort(axes.begin(), axes.end());

            if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
                ORMI_THROW(ShapeError, "Reduction axes must be unique.");
            }

            return axes;
        }

        Shape reduced_shape(
            const Shape& input,
            const std::vector<std::int64_t>& axes,
            bool keepdims
        ) {
            std::vector<std::int64_t> dims;
            dims.reserve(keepdims ? input.rank() : input.rank() - axes.size());

            for (std::size_t axis = 0; axis < input.rank(); ++axis) {
                const bool reduced = std::binary_search(
                    axes.begin(),
                    axes.end(),
                    static_cast<std::int64_t>(axis)
                );

                if (reduced) {
                    if (keepdims) {
                        dims.push_back(1);
                    }
                } else {
                    dims.push_back(input[axis]);
                }
            }

            return Shape(std::move(dims));
        }
    }

    std::shared_ptr<Tensor> all_finite(const std::shared_ptr<Tensor>& input) {
        if (!input) {
            ORMI_THROW(OperationError, "All-finite reduction requires a tensor.");
        }

        if (!is_floating(input->dtype())) {
            ORMI_THROW(DTypeError, "All-finite reduction supports floating-point tensors only.");
        }

        return std::make_shared<Tensor>(
            op_ref<all_finite_op>(),
            std::vector<std::shared_ptr<Tensor>>{input},
            input->device(),
            Shape{},
            DType::boolean
        );
    }

    std::shared_ptr<Tensor> sum(
        const std::shared_ptr<Tensor>& input,
        std::vector<std::int64_t> axes,
        bool keepdims
    ) {
        if (!input) {
            ORMI_THROW(OperationError, "Sum requires a tensor.");
        }

        if (!is_numeric(input->dtype())) {
            ORMI_THROW(DTypeError, "Sum requires a numeric tensor.");
        }

        axes = normalize_axes(std::move(axes), input->rank());
        auto shape = reduced_shape(input->shape(), axes, keepdims);

        if (!precision::state.enabled) {
            return std::make_shared<Tensor>(
                op_ref<sum_op>(),
                Reduce{std::move(axes), keepdims},
                std::vector<std::shared_ptr<Tensor>>{input},
                input->device(),
                std::move(shape),
                input->dtype()
            );
        }

        const auto dtype = precision::apply(sum_op.precision, input->dtype());
        const auto parent = input->dtype() == dtype ? input : cast(input, dtype);

        return std::make_shared<Tensor>(
            op_ref<sum_op>(),
            Reduce{std::move(axes), keepdims},
            std::vector<std::shared_ptr<Tensor>>{parent},
            input->device(),
            std::move(shape),
            dtype
        );
    }

    std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& input, bool keepdims) {
        return sum(input, std::vector<std::int64_t>{}, keepdims);
    }

    std::shared_ptr<Tensor> sum(
        const std::shared_ptr<Tensor>& input,
        std::initializer_list<std::int64_t> axes,
        bool keepdims
    ) {
        return sum(input, std::vector<std::int64_t>(axes), keepdims);
    }

    bool is_sum(const Tensor& tensor) {
        return tensor.op() == &sum_op;
    }

    std::shared_ptr<Tensor> mean(
        const std::shared_ptr<Tensor>& input,
        std::vector<std::int64_t> axes,
        bool keepdims
    ) {
        if (!input) {
            ORMI_THROW(OperationError, "Mean requires a tensor.");
        }

        if (!is_floating(input->dtype())) {
            ORMI_THROW(DTypeError, "Mean requires a floating-point tensor.");
        }

        axes = normalize_axes(std::move(axes), input->rank());
        auto shape = reduced_shape(input->shape(), axes, keepdims);
        const auto dtype = precision::apply(mean_op.precision, input->dtype());
        const auto parent = input->dtype() == dtype ? input : cast(input, dtype);

        return std::make_shared<Tensor>(
            op_ref<mean_op>(),
            Reduce{std::move(axes), keepdims},
            std::vector<std::shared_ptr<Tensor>>{parent},
            input->device(),
            std::move(shape),
            dtype
        );
    }

    std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& input, bool keepdims) {
        return mean(input, std::vector<std::int64_t>{}, keepdims);
    }

    std::shared_ptr<Tensor> mean(
        const std::shared_ptr<Tensor>& input,
        std::initializer_list<std::int64_t> axes,
        bool keepdims
    ) {
        return mean(input, std::vector<std::int64_t>(axes), keepdims);
    }

    bool is_mean(const Tensor& tensor) {
        return tensor.op() == &mean_op;
    }
}
