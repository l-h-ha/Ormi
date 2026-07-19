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
#include <ormi/core/ops/kind.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/runtime/dispatch/reduction.h>
#include <ormi/core/runtime/dispatch/finite.h>
#include <ormi/core/runtime/dispatch/memory.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

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

        void adjoint_sum(
            const std::shared_ptr<Tensor>& node,
            const std::shared_ptr<Tensor>& upstream,
            Grads& gradients
        ) {
            const auto& input = node->parents()[0];
            const auto& attrs = std::get<Reduce>(node->attrs());
            auto gradient = upstream;

            if (!attrs.keepdims) {
                auto dims = input->shape().dims();

                for (const auto axis : attrs.axes) {
                    dims[static_cast<std::size_t>(axis)] = 1;
                }

                gradient = reshape(gradient, Shape(std::move(dims)));
            }

            gradients.add(input, expand(gradient, input->shape()));
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
            all_finite_op,
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
                sum_op,
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
            sum_op,
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
}
