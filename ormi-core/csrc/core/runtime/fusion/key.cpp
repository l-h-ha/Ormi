#include <cstdint>
#include <type_traits>
#include <variant>

#include "core/runtime/fusion/key.h"

namespace ormi::core::runtime::fusion {
    namespace {
        std::uint64_t combine(std::uint64_t hash, std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
            return hash;
        }
    }

    Key::Key(const Expr& expression)
        : dtype_(expression.dtype),
          shape_(expression.shape),
          nodes_(expression.nodes),
          output_(expression.output),
          auxiliary_outputs_(expression.auxiliary_outputs),
          reduction_(expression.reduction),
          hash_(hash(expression)) {}

    bool Key::matches(const Expr& expression) const {
        return
            dtype_ == expression.dtype &&
            shape_ == expression.shape &&
            nodes_ == expression.nodes &&
            output_ == expression.output &&
            auxiliary_outputs_ == expression.auxiliary_outputs &&
            reduction_ == expression.reduction;
    }

    std::uint64_t Key::hash() const noexcept {
        return hash_;
    }

    std::uint64_t Key::hash(const Expr& expression) {
        auto result = combine(
            14695981039346656037ull,
            static_cast<std::uint64_t>(expression.dtype)
        );
        result = combine(result, expression.output);
        result = combine(result, expression.auxiliary_outputs.size());
        for (const auto output : expression.auxiliary_outputs) {
            result = combine(result, output);
        }
        result = combine(result, expression.nodes.size());
        result = combine(result, expression.reduction.has_value());
        result = combine(result, expression.shape.size());
        for (const auto extent : expression.shape) {
            result = combine(result, extent);
        }

        if (expression.reduction) {
            result = combine(
                result,
                static_cast<std::uint64_t>(expression.reduction->op)
            );
            result = combine(result, expression.reduction->output.size);
            for (const auto& axis : expression.reduction->output.axes) {
                result = combine(result, axis.divisor);
                result = combine(result, axis.extent);
                result = combine(result, axis.stride);
            }
            result = combine(result, expression.reduction->reduced.size);
            for (const auto& axis : expression.reduction->reduced.axes) {
                result = combine(result, axis.divisor);
                result = combine(result, axis.extent);
                result = combine(result, axis.stride);
            }
        }

        for (const auto& node : expression.nodes) {
            result = combine(
                result,
                static_cast<std::uint64_t>(node.dtype)
            );
            result = combine(result, node.instruction.index());
            std::visit(
                [&result](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, Input>) {
                        result = combine(result, value.index);
                        result = combine(result, value.linear);
                        for (const auto& axis : value.axes) {
                            result = combine(result, axis.divisor);
                            result = combine(result, axis.extent);
                            result = combine(result, axis.stride);
                        }
                    } else if constexpr (std::is_same_v<T, Constant>) {
                        result = combine(
                            result,
                            static_cast<std::uint64_t>(value)
                        );
                    } else if constexpr (std::is_same_v<T, Cast>) {
                        result = combine(result, value.input);
                    } else if constexpr (std::is_same_v<T, Unary>) {
                        result = combine(
                            result,
                            static_cast<std::uint64_t>(value.op)
                        );
                        result = combine(result, value.input);
                    } else {
                        result = combine(
                            result,
                            static_cast<std::uint64_t>(value.op)
                        );
                        result = combine(result, value.lhs);
                        result = combine(result, value.rhs);
                    }
                },
                node.instruction
            );
        }

        return result;
    }
}
