#include <algorithm>
#include <array>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/attrs.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"

namespace ormi::core::runtime::fusion {
    namespace {
        bool supported_dtype(DType dtype) {
            return
                dtype == DType::float64 ||
                dtype == DType::float32 ||
                dtype == DType::float16;
        }

        bool scalar(const Tensor& tensor) {
            return tensor.op() && !std::holds_alternative<std::monostate>(tensor.op()->scalar);
        }

        bool operation(const Tensor& tensor) {
            return scalar(tensor) || ops::is_cast(tensor);
        }

        bool is_reduction(const Tensor& tensor) {
            return ops::is_sum(tensor) || ops::is_mean(tensor);
        }

        std::vector<std::size_t> dimensions(const Tensor& tensor) {
            std::vector<std::size_t> result;
            result.reserve(tensor.rank());
            for (std::size_t dimension = 0; dimension < tensor.rank(); ++dimension) {
                result.push_back(
                    static_cast<std::size_t>(tensor.shape()[dimension])
                );
            }
            return result;
        }

        Tensor* pointwise_root(Tensor& output) {
            return is_reduction(output) ? output.parents()[0].get() : &output;
        }

        Iteration iteration(
            const Tensor& input,
            const std::vector<std::int64_t>& reduced_axes,
            bool reduced
        ) {
            std::vector<std::size_t> selected;
            selected.reserve(input.rank());

            for (std::size_t axis = 0; axis < input.rank(); ++axis) {
                const auto is_reduced = std::binary_search(
                    reduced_axes.begin(),
                    reduced_axes.end(),
                    static_cast<std::int64_t>(axis)
                );
                if (is_reduced == reduced) {
                    selected.push_back(axis);
                }
            }

            std::size_t size = 1;
            for (const auto axis : selected) {
                size *= static_cast<std::size_t>(input.shape()[axis]);
            }

            if (size == 0) {
                return Iteration{0, {}};
            }

            const auto source_strides = contiguous_strides(input.shape());
            std::vector<Axis> axes(selected.size());
            std::size_t divisor = 1;

            for (std::size_t i = selected.size(); i > 0; --i) {
                const auto axis = selected[i - 1];
                const auto extent = static_cast<std::size_t>(input.shape()[axis]);
                axes[i - 1] = Axis{
                    divisor,
                    extent,
                    static_cast<std::size_t>(source_strides[axis])
                };
                divisor *= extent;
            }
            return Iteration{size, std::move(axes)};
        }

        std::optional<Reduction> reduction(const Tensor& output) {
            if (!is_reduction(output)) {
                return std::nullopt;
            }

            const auto& attrs = std::get<Reduce>(output.attrs());
            const auto& input = *output.parents()[0];
            return Reduction{
                ops::is_mean(output) ? ops::ReduceOp::mean : ops::ReduceOp::sum,
                iteration(input, attrs.axes, false),
                iteration(input, attrs.axes, true)
            };
        }

        class Builder {
        public:
            explicit Builder(
                Tensor& root,
                std::span<Tensor* const> auxiliary_outputs = {},
                const std::unordered_set<const Tensor*>* boundaries = nullptr
            )
                : root_(pointwise_root(root)),
                  auxiliary_outputs_(auxiliary_outputs),
                  boundaries_(boundaries),
                  expr_{
                      root_->dtype(),
                      dimensions(*root_),
                      {},
                      {},
                      0,
                      {},
                      0,
                      root_->numel(),
                      reduction(root)
                  }
            {
                expr_.nodes.reserve(initial_capacity);
                expr_.inputs.reserve(initial_capacity);
                indices_.reserve(initial_capacity);
            }

            std::optional<Expr> run() {
                std::pmr::vector<std::pair<Tensor*, bool>> stack{&scratch_};
                stack.reserve(initial_capacity);
                for (auto output = auxiliary_outputs_.rbegin();
                     output != auxiliary_outputs_.rend();
                     ++output) {
                    stack.emplace_back(*output, false);
                }
                stack.emplace_back(root_, false);

                while (!stack.empty()) {
                    auto [tensor, expanded] = std::move(stack.back());
                    stack.pop_back();

                    if (indices_.contains(tensor)) {
                        continue;
                    }

                    if (constant(*tensor)) {
                        append(
                            tensor,
                            ops::is_zero(*tensor) ? Constant::zero : Constant::one
                        );
                        continue;
                    }

                    if (boundary(*tensor)) {
                        if (!append_input(tensor)) {
                            return std::nullopt;
                        }
                        continue;
                    }

                    if (expanded) {
                        append_operation(*tensor);
                        continue;
                    }

                    stack.emplace_back(tensor, true);

                    for (auto parent = tensor->parents().rbegin(); parent != tensor->parents().rend(); ++parent) {
                        if (!indices_.contains(parent->get())) {
                            stack.emplace_back(parent->get(), false);
                        }
                    }
                }

                if (expr_.operations < 2) {
                    return std::nullopt;
                }

                expr_.output = indices_.at(root_);
                expr_.auxiliary_outputs.reserve(auxiliary_outputs_.size());
                for (const auto* output : auxiliary_outputs_) {
                    expr_.auxiliary_outputs.push_back(indices_.at(output));
                }
                return std::move(expr_);
            }

        private:
            static constexpr std::size_t initial_capacity = 8;
            static constexpr std::size_t initial_scratch_bytes = 1024;

            bool constant(const Tensor& tensor) const {
                return
                    supported_dtype(tensor.dtype()) &&
                    (ops::is_zero(tensor) || ops::is_one(tensor));
            }

            bool boundary(const Tensor& tensor) const {
                return
                    tensor.realized() ||
                    (
                        boundaries_ &&
                        boundaries_->contains(&tensor)
                    ) ||
                    !operation(tensor) ||
                    !supported_dtype(tensor.dtype());
            }

            bool append_input(Tensor* tensor) {
                if (
                    !supported_dtype(tensor->dtype()) ||
                    tensor->rank() > root_->rank()
                ) {
                    return false;
                }

                const auto input = expr_.inputs.size();
                const auto linear =
                    tensor->shape() == root_->shape() && tensor->is_contiguous();
                std::vector<Axis> axes;

                if (!linear && !root_->shape().empty()) {
                    const auto output_strides = contiguous_strides(root_->shape());
                    const auto input_strides = tensor->strides();
                    const auto rank_offset = root_->rank() - tensor->rank();
                    axes.reserve(tensor->rank());

                    for (std::size_t axis = rank_offset; axis < root_->rank(); ++axis) {
                        const auto input_axis = axis - rank_offset;
                        const auto input_extent = tensor->shape()[input_axis];
                        const auto output_extent = root_->shape()[axis];

                        if (input_extent != 1 && input_extent != output_extent) {
                            return false;
                        }

                        if (input_extent != 1 && input_strides[input_axis] != 0) {
                            axes.push_back(Axis{
                                static_cast<std::size_t>(output_strides[axis]),
                                static_cast<std::size_t>(output_extent),
                                static_cast<std::size_t>(input_strides[input_axis])
                            });
                        }
                    }
                }

                expr_.inputs.push_back(tensor);
                append(tensor, Input{input, std::move(axes), linear});
                return true;
            }

            void append_operation(const Tensor& tensor) {
                if (ops::is_cast(tensor)) {
                    append(&tensor, Cast{indices_.at(tensor.parents()[0].get())});
                    ++expr_.operations;
                    return;
                }

                if (const auto* unary = std::get_if<ops::UnaryOp>(&tensor.op()->scalar)) {
                    append(
                        &tensor,
                        Unary{*unary, indices_.at(tensor.parents()[0].get())}
                    );
                    ++expr_.operations;
                    return;
                }

                if (const auto* binary = std::get_if<ops::BinaryOp>(&tensor.op()->scalar)) {
                    append(
                        &tensor,
                        Binary{
                            *binary,
                            indices_.at(tensor.parents()[0].get()),
                            indices_.at(tensor.parents()[1].get())
                        }
                    );
                    ++expr_.operations;
                    return;
                }

                ORMI_THROW(OperationError, "Fusion encountered an invalid scalar operation.");
            }

            void append(const Tensor* tensor, Instruction instruction) {
                if (expr_.nodes.size() > static_cast<std::size_t>(UINT32_MAX)) {
                    ORMI_THROW(OperationError, "Fusion expression exceeds its node index range.");
                }

                const auto index = static_cast<Index>(expr_.nodes.size());
                expr_.nodes.push_back(Node{tensor->dtype(), std::move(instruction)});
                indices_.emplace(tensor, index);
            }

            Tensor* root_;
            std::span<Tensor* const> auxiliary_outputs_;
            const std::unordered_set<const Tensor*>* boundaries_;
            Expr expr_;
            std::array<std::byte, initial_scratch_bytes> scratch_storage_;
            std::pmr::monotonic_buffer_resource scratch_{
                scratch_storage_.data(),
                scratch_storage_.size()
            };
            std::pmr::unordered_map<const Tensor*, Index> indices_{&scratch_};
        };
    }

    bool eligible(Tensor& output) {
        const auto* root = pointwise_root(output);
        return
            !output.realized() &&
            supported_dtype(root->dtype()) &&
            root->is_contiguous() &&
            operation(*root);
    }

    bool groupable(Tensor& output) {
        return eligible(output) && !is_reduction(output);
    }

    namespace {
        std::optional<Expr> build_group(
            Tensor& root,
            std::span<Tensor* const> auxiliary_outputs,
            const std::unordered_set<const Tensor*>* boundaries
        ) {
            if (!eligible(root) || (is_reduction(root) && !auxiliary_outputs.empty())) {
                return std::nullopt;
            }

            std::unordered_set<const Tensor*> outputs{&root};
            for (auto* output : auxiliary_outputs) {
                if (
                    !output ||
                    is_reduction(*output) ||
                    !outputs.insert(output).second ||
                    !eligible(*output) ||
                    output->shape() != root.shape() ||
                    output->dtype() != root.dtype() ||
                    output->device().backend_id() != root.device().backend_id() ||
                    output->device().index() != root.device().index()
                ) {
                    return std::nullopt;
                }
            }
            return Builder(root, auxiliary_outputs, boundaries).run();
        }
    }

    std::optional<Expr> build(Tensor& root) {
        if (!eligible(root)) {
            return std::nullopt;
        }
        return Builder(root).run();
    }

    std::optional<Expr> build(
        Tensor& root,
        const std::unordered_set<const Tensor*>& boundaries
    ) {
        if (!eligible(root)) {
            return std::nullopt;
        }
        return Builder(root, {}, &boundaries).run();
    }

    std::optional<Expr> build(
        Tensor& root,
        std::span<Tensor* const> auxiliary_outputs
    ) {
        return build_group(root, auxiliary_outputs, nullptr);
    }

    std::optional<Expr> build(
        Tensor& root,
        std::span<Tensor* const> auxiliary_outputs,
        const std::unordered_set<const Tensor*>& boundaries
    ) {
        return build_group(root, auxiliary_outputs, &boundaries);
    }
}
