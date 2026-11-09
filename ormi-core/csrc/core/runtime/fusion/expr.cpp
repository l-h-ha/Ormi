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
#include "core/runtime/fusion/layout.h"

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

        Tensor* pointwise_root(Tensor& output) {
            return is_reduction(output) ? output.parents()[0].get() : &output;
        }

        bool eligible(Tensor& output) {
            const auto* root = pointwise_root(output);
            return
                !output.realized() &&
                supported_dtype(root->dtype()) &&
                root->is_contiguous() &&
                operation(*root);
        }

        std::optional<Reduction> reduction(const Tensor& output) {
            if (!is_reduction(output)) {
                return std::nullopt;
            }

            const auto& attrs = std::get<Reduce>(output.attrs());
            const auto& input = *output.parents()[0];
            return layout::reduction(
                input.shape(),
                attrs.axes,
                ops::is_mean(output) ? ops::ReduceOp::mean : ops::ReduceOp::sum
            );
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
                      layout::dimensions(root_->shape()),
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
                std::pmr::vector<Frame> stack{&scratch_};
                stack.reserve(initial_capacity);
                for (auto output = auxiliary_outputs_.rbegin();
                     output != auxiliary_outputs_.rend();
                     ++output) {
                    stack.emplace_back(*output, Phase::visit);
                }
                stack.emplace_back(root_, Phase::visit);

                while (!stack.empty()) {
                    const auto frame = stack.back();
                    stack.pop_back();
                    auto* tensor = frame.tensor;

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

                    if (frame.phase == Phase::visit) {
                        stack.emplace_back(tensor, Phase::append);

                        for (auto parent = tensor->parents().rbegin(); parent != tensor->parents().rend(); ++parent) {
                            if (!indices_.contains(parent->get())) {
                                stack.emplace_back(parent->get(), Phase::visit);
                            }
                        }
                        continue;
                    }

                    append_operation(*tensor);
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
            enum class Phase : std::uint8_t {
                visit,
                append
            };

            struct Frame {
                Tensor* tensor;
                Phase phase;

                Frame(Tensor* tensor, Phase phase) : tensor(tensor), phase(phase) {}
            };

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

                const auto index = expr_.inputs.size();
                const auto linear =
                    tensor->shape() == root_->shape() && tensor->is_contiguous();
                std::vector<Axis> axes;
                if (
                    !linear && !layout::input_axes(
                        tensor->shape(),
                        tensor->strides(),
                        root_->shape(),
                        axes
                    )
                ) {
                    return false;
                }

                expr_.inputs.push_back(tensor);
                append(tensor, Input{index, std::move(axes), linear});
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
                    output->device() != root.device()
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
