#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <ormi/backend/fusion.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>

#include "backends/cuda/fusion/codegen/emit.h"
#include "backends/cuda/fusion/tile.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        std::string unary(core::DType dtype, core::ops::UnaryOp op, std::string_view input) {
            const auto x = std::string(input);
            const auto zero = "decltype(" + x + ")(0)";
            const auto one = "decltype(" + x + ")(1)";
            const bool single = dtype != core::DType::float64;
            const std::string exp = single ? "expf" : "exp";
            const std::string log1p = single ? "log1pf" : "log1p";

            switch (op) {
                case core::ops::UnaryOp::neg:
                    return "-" + x;
                case core::ops::UnaryOp::exp:
                    return exp + "(" + x + ")";
                case core::ops::UnaryOp::log:
                    return std::string(single ? "logf" : "log") + "(" + x + ")";
                case core::ops::UnaryOp::sin:
                    return std::string(single ? "sinf" : "sin") + "(" + x + ")";
                case core::ops::UnaryOp::cos:
                    return std::string(single ? "cosf" : "cos") + "(" + x + ")";
                case core::ops::UnaryOp::abs:
                    return std::string(single ? "fabsf" : "fabs") + "(" + x + ")";
                case core::ops::UnaryOp::tanh:
                    return std::string(single ? "tanhf" : "tanh") + "(" + x + ")";
                case core::ops::UnaryOp::erf:
                    return std::string(single ? "erff" : "erf") + "(" + x + ")";
                case core::ops::UnaryOp::log1p:
                    return log1p + "(" + x + ")";
                case core::ops::UnaryOp::expm1:
                    return std::string(single ? "expm1f" : "expm1") + "(" + x + ")";
                case core::ops::UnaryOp::rsqrt:
                    return std::string(single ? "rsqrtf" : "rsqrt") + "(" + x + ")";
                case core::ops::UnaryOp::relu:
                    return x + " < " + zero + " ? " + zero + " : " + x;
                case core::ops::UnaryOp::positive:
                    return x + " != " + x + " ? " + x + " : (" + x + " > " + zero + " ? " + one + " : " +
                           zero + ")";
                case core::ops::UnaryOp::sign:
                    return x + " != " + x + " ? " + x + " : (" + x + " > " + zero + " ? " + one + " : (" + x +
                           " < " + zero + " ? -" + one + " : " + zero + "))";
                case core::ops::UnaryOp::sigmoid:
                    return x + " >= " + zero + " ? " + one + " / (" + one + " + " + exp + "(-" + x +
                           ")) : " + exp + "(" + x + ") / (" + one + " + " + exp + "(" + x + "))";
                case core::ops::UnaryOp::softplus:
                    return x + " > " + zero + " ? " + x + " + " + log1p + "(" + exp + "(-" + x +
                           ")) : " + log1p + "(" + exp + "(" + x + "))";
            }
            ORMI_THROW(OperationError, "CUDA fusion received an unknown unary operation.");
        }

        std::string binary(core::ops::BinaryOp op, std::string_view lhs, std::string_view rhs) {
            const auto left = std::string(lhs);
            const auto right = std::string(rhs);
            switch (op) {
                case core::ops::BinaryOp::add:
                    return left + " + " + right;
                case core::ops::BinaryOp::sub:
                    return left + " - " + right;
                case core::ops::BinaryOp::mul:
                    return left + " * " + right;
                case core::ops::BinaryOp::div:
                    return left + " / " + right;
                case core::ops::BinaryOp::minimum:
                case core::ops::BinaryOp::maximum: {
                    const auto comparison = op == core::ops::BinaryOp::minimum ? " < " : " > ";
                    return left + " != " + left + " ? " + left + " : (" + right + " != " + right + " ? " +
                           right + " : (" + left + comparison + right + " ? " + left + " : " + right + "))";
                }
            }
            ORMI_THROW(OperationError, "CUDA fusion received an unknown binary operation.");
        }

        std::string rounded(core::DType dtype, std::string expression) {
            if (dtype == core::DType::float16) {
                return "ormi_half_round(" + expression + ")";
            }
            return expression;
        }

        std::string input_index(
            const core::runtime::fusion::Input& input,
            const Tile* tile,
            bool physical_order
        ) {
            std::string index;
            if (tile) {
                const auto& strides = tile->inputs[input.index];
                for (std::size_t dimension = 0; dimension < strides.size(); ++dimension) {
                    if (strides[dimension] == 0) {
                        continue;
                    }
                    if (!index.empty()) {
                        index += " + ";
                    }
                    index += "index" + std::to_string(dimension);
                    if (strides[dimension] != 1) {
                        index += " * " + std::to_string(strides[dimension]);
                    }
                }
                if (index.empty()) {
                    index = "0";
                }
            } else if (physical_order) {
                index = "reduced_index";
            } else {
                index = input.linear ? "i" : offset("i", input.axes);
            }
            return index;
        }

        std::string input_value(
            core::DType dtype,
            const core::runtime::fusion::Input& input,
            const Tile* tile,
            bool physical_order
        ) {
            const auto index = input_index(input, tile, physical_order);
            auto expression = "input" + std::to_string(input.index) + "[" + index + "]";
            if (dtype == core::DType::float16) {
                return "ormi_half_value(" + expression + ")";
            }
            return expression;
        }

        std::string cast_value(core::DType dtype, std::string_view expression) {
            switch (dtype) {
                case core::DType::float64:
                    return "static_cast<double>(" + std::string(expression) + ")";
                case core::DType::float32:
                    return "static_cast<float>(" + std::string(expression) + ")";
                case core::DType::float16:
                    return "ormi_half_round(" + std::string(expression) + ")";
                case core::DType::int32:
                case core::DType::boolean:
                    ORMI_THROW(DTypeError, "CUDA fusion cannot cast to a non-floating dtype.");
            }

            ORMI_THROW(DTypeError, "CUDA fusion received an unknown cast dtype.");
        }

        void require_prior(core::runtime::fusion::Index referenced, std::size_t current) {
            if (referenced >= current) {
                ORMI_THROW(OperationError, "Fusion expression nodes must reference prior nodes.");
            }
        }
    }

    std::string offset(std::string_view index, const std::vector<core::runtime::fusion::Axis>& axes) {
        if (axes.empty()) {
            return "0";
        }

        std::ostringstream result;
        for (std::size_t i = 0; i < axes.size(); ++i) {
            const auto& axis = axes[i];
            if (i != 0) {
                result << " + ";
            }

            result << "(";
            if (axis.divisor != 1) {
                result << "(" << index << " / " << axis.divisor << ")";
            } else {
                result << index;
            }
            result << " % " << axis.extent << ")";
            if (axis.stride != 1) {
                result << " * " << axis.stride;
            }
        }
        return std::move(result).str();
    }

    void emit_nodes(
        std::ostringstream& source,
        const core::runtime::fusion::Expr& expression,
        const Tile* tile,
        bool physical_order
    ) {
        for (std::size_t i = 0; i < expression.nodes.size(); ++i) {
            const auto& node = expression.nodes[i];
            source << "        const " << value_type(node.dtype) << " "
                   << value(static_cast<core::runtime::fusion::Index>(i)) << " = ";
            const auto& instruction = node.instruction;

            if (const auto* input = std::get_if<core::runtime::fusion::Input>(&instruction)) {
                if (input->index >= expression.inputs.size()) {
                    ORMI_THROW(OperationError, "Fusion expression references an invalid input.");
                }
                source << input_value(node.dtype, *input, tile, physical_order);
            } else if (const auto* constant = std::get_if<core::runtime::fusion::Constant>(&instruction)) {
                source << literal(node.dtype, *constant);
            } else if (const auto* cast_node = std::get_if<core::runtime::fusion::Cast>(&instruction)) {
                require_prior(cast_node->input, i);
                source << cast_value(node.dtype, value(cast_node->input));
            } else if (const auto* unary_node = std::get_if<core::runtime::fusion::Unary>(&instruction)) {
                require_prior(unary_node->input, i);
                const auto operation = unary(node.dtype, unary_node->op, value(unary_node->input));
                source
                    << (i == expression.output && !expression.reduction ? operation
                                                                        : rounded(node.dtype, operation));
            } else if (const auto* binary_node = std::get_if<core::runtime::fusion::Binary>(&instruction)) {
                require_prior(binary_node->lhs, i);
                require_prior(binary_node->rhs, i);
                const auto operation =
                    binary(binary_node->op, value(binary_node->lhs), value(binary_node->rhs));
                source
                    << (i == expression.output && !expression.reduction ? operation
                                                                        : rounded(node.dtype, operation));
            }

            source << ";\n";
        }
    }

}
