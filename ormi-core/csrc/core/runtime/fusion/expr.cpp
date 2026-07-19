#include <cstdint>
#include <memory_resource>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"

namespace ormi::core::runtime::fusion {
    namespace {
        bool supported_dtype(DType dtype) {
            return dtype == DType::float64 || dtype == DType::float32;
        }

        bool scalar(const Tensor& tensor) {
            return tensor.op() && !std::holds_alternative<std::monostate>(tensor.op()->scalar);
        }

        class Builder {
        public:
            explicit Builder(Tensor& root)
                : root_(&root), expr_{root.dtype(), {}, {}, 0, 0}
            {
                expr_.nodes.reserve(initial_capacity);
                expr_.inputs.reserve(initial_capacity);
                indices_.reserve(initial_capacity);
            }

            std::optional<Expr> run() {
                if (!eligible_root()) {
                    return std::nullopt;
                }

                std::pmr::vector<std::pair<Tensor*, bool>> stack{&scratch_};
                stack.reserve(initial_capacity);
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
                return std::move(expr_);
            }

        private:
            static constexpr std::size_t initial_capacity = 8;

            bool eligible_root() const {
                return
                    !root_->realized() &&
                    supported_dtype(root_->dtype()) &&
                    root_->is_contiguous() &&
                    scalar(*root_);
            }

            bool constant(const Tensor& tensor) const {
                return
                    tensor.dtype() == expr_.dtype &&
                    tensor.shape() == root_->shape() &&
                    (ops::is_zero(tensor) || ops::is_one(tensor));
            }

            bool boundary(const Tensor& tensor) const {
                return
                    tensor.realized() ||
                    !scalar(tensor) ||
                    tensor.dtype() != expr_.dtype ||
                    tensor.shape() != root_->shape() ||
                    !tensor.is_contiguous();
            }

            bool append_input(Tensor* tensor) {
                if (
                    tensor->dtype() != expr_.dtype ||
                    tensor->shape() != root_->shape() ||
                    !tensor->is_contiguous()
                ) {
                    return false;
                }

                const auto input = expr_.inputs.size();
                expr_.inputs.push_back(tensor);
                append(tensor, Input{input});
                return true;
            }

            void append_operation(const Tensor& tensor) {
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

            void append(const Tensor* tensor, Node node) {
                if (expr_.nodes.size() > static_cast<std::size_t>(UINT32_MAX)) {
                    ORMI_THROW(OperationError, "Fusion expression exceeds its node index range.");
                }

                const auto index = static_cast<Index>(expr_.nodes.size());
                expr_.nodes.push_back(std::move(node));
                indices_.emplace(tensor, index);
            }

            Tensor* root_;
            Expr expr_;
            std::pmr::monotonic_buffer_resource scratch_;
            std::pmr::unordered_map<const Tensor*, Index> indices_{&scratch_};
        };
    }

    std::optional<Expr> build(Tensor& root) {
        return Builder(root).run();
    }
}
