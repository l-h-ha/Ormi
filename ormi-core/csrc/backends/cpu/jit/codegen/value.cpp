#include <cstddef>
#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>

#include "backends/cpu/jit/codegen/code.h"
#include "backends/cpu/jit/math.h"

namespace ormi::backends::cpu::jit::codegen {
    namespace {
        bool is_output(
            const core::runtime::fusion::Expr& expression,
            std::size_t index
        ) {
            return
                index == expression.output ||
                std::find(
                    expression.auxiliary_outputs.begin(),
                    expression.auxiliary_outputs.end(),
                    index
                ) != expression.auxiliary_outputs.end();
        }

        std::string unary(
            core::ops::UnaryOp op,
            std::string_view input
        ) {
            switch (op) {
                case core::ops::UnaryOp::neg:
                    return "-" + std::string(input);
                case core::ops::UnaryOp::exp:
                    return "std::exp(" + std::string(input) + ")";
                case core::ops::UnaryOp::log:
                    return "std::log(" + std::string(input) + ")";
                case core::ops::UnaryOp::sin:
                    return "std::sin(" + std::string(input) + ")";
                case core::ops::UnaryOp::cos:
                    return "std::cos(" + std::string(input) + ")";
            }

            ORMI_THROW(
                OperationError,
                "CPU fusion code generation received an unknown unary operation."
            );
        }

        std::string_view binary(core::ops::BinaryOp op) {
            switch (op) {
                case core::ops::BinaryOp::add:
                    return "+";
                case core::ops::BinaryOp::sub:
                    return "-";
                case core::ops::BinaryOp::mul:
                    return "*";
                case core::ops::BinaryOp::div:
                    return "/";
            }

            ORMI_THROW(
                OperationError,
                "CPU fusion code generation received an unknown binary operation."
            );
        }

        std::string rounded(core::DType dtype, std::string expression) {
            if (dtype == core::DType::float16) {
                return "ormi_half_round(" + expression + ")";
            }
            return expression;
        }

        std::string input_offset(
            const core::runtime::fusion::Input& input
        ) {
            if (input.linear) {
                return "i";
            }
            return offset("i", input.axes);
        }

        std::string input_value(
            core::DType dtype,
            const core::runtime::fusion::Input& input,
            const Loop& loop
        ) {
            auto expression = "input" + std::to_string(input.index);
            if (loop.strided) {
                expression += "[";
                bool first = true;
                for (
                    std::size_t dimension = 0;
                    dimension < loop.strides[input.index].size();
                    ++dimension
                ) {
                    const auto stride =
                        loop.strides[input.index][dimension];
                    if (stride == 0) {
                        continue;
                    }
                    if (!first) {
                        expression += " + ";
                    }
                    first = false;
                    expression += "index" + std::to_string(dimension);
                    if (stride != 1) {
                        expression += " * " + std::to_string(stride);
                    }
                }
                expression += first ? "0]" : "]";
                if (dtype == core::DType::float16) {
                    return "ormi_half_value(" + expression + ")";
                }
                return expression;
            }

            const auto access = loop.vectorized
                ? loop.inputs[input.index]
                : Access::indexed;

            switch (access) {
                case Access::linear:
                    expression += "[i]";
                    break;
                case Access::scalar:
                    expression += "_scalar";
                    break;
                case Access::span:
                    expression += "[input" +
                        std::to_string(input.index) + "_base + j]";
                    break;
                case Access::fixed:
                    expression += "_fixed";
                    break;
                case Access::indexed:
                    expression += "[" + input_offset(input) + "]";
                    break;
            }
            if (dtype == core::DType::float16) {
                return "ormi_half_value(" + expression + ")";
            }
            return expression;
        }

        std::string cast_value(
            core::DType dtype,
            std::string_view expression
        ) {
            switch (dtype) {
                case core::DType::float64:
                    return "static_cast<double>(" +
                        std::string(expression) + ")";
                case core::DType::float32:
                    return "static_cast<float>(" +
                        std::string(expression) + ")";
                case core::DType::float16:
                    return "ormi_half_round(" +
                        std::string(expression) + ")";
                case core::DType::int32:
                case core::DType::boolean:
                    ORMI_THROW(
                        DTypeError,
                        "CPU fusion cannot cast to a non-floating dtype."
                    );
            }

            ORMI_THROW(
                DTypeError,
                "CPU fusion received an unknown cast dtype."
            );
        }

        void require_prior(
            core::runtime::fusion::Index referenced,
            std::size_t current
        ) {
            if (referenced >= current) {
                ORMI_THROW(
                    OperationError,
                    "Fusion expression nodes must reference prior nodes."
                );
            }
        }

        std::string packed_strided_input(
            const core::runtime::fusion::Input& input,
            const Loop& loop
        ) {
            const auto name = "input" + std::to_string(input.index);
            const auto& strides = loop.strides[input.index];
            const auto inner = loop.reduced_dimensions.back();
            std::string address;

            for (
                std::size_t dimension = 0;
                dimension < strides.size();
                ++dimension
            ) {
                const auto stride = strides[dimension];
                if (stride == 0) {
                    continue;
                }
                if (!address.empty()) {
                    address += " + ";
                }
                address += dimension == inner
                    ? "inner_index"
                    : "index" + std::to_string(dimension);
                if (stride != 1) {
                    address += " * " + std::to_string(stride);
                }
            }

            if (strides[inner] == 0) {
                return "_mm256_set1_ps(" + name + "[" +
                    (address.empty() ? "0" : address) + "])";
            }
            if (strides[inner] == 1) {
                return "_mm256_loadu_ps(" + name + " + " + address + ")";
            }

            ORMI_THROW(
                OperationError,
                "Packed strided fusion requires unit or broadcast inner strides."
            );
        }

        std::string packed_input(
            const core::runtime::fusion::Input& input,
            core::DType dtype,
            const Loop& loop
        ) {
            if (loop.strided) {
                if (dtype != core::DType::float32) {
                    ORMI_THROW(
                        OperationError,
                        "Packed strided fusion requires Float32 inputs."
                    );
                }
                return packed_strided_input(input, loop);
            }

            const auto name = "input" + std::to_string(input.index);
            const bool float16 = dtype == core::DType::float16;
            switch (loop.inputs[input.index]) {
                case Access::linear:
                    return float16
                        ? "ormi_half_load8(" + name + " + i)"
                        : "_mm256_loadu_ps(" + name + " + i)";
                case Access::scalar:
                    return "_mm256_set1_ps(" + (float16
                        ? "ormi_half_value(" + name + "_scalar)"
                        : name + "_scalar") + ")";
                case Access::span:
                    return float16
                        ? "ormi_half_load8(" + name + " + input" +
                            std::to_string(input.index) + "_base + j)"
                        : "_mm256_loadu_ps(" + name + " + input" +
                            std::to_string(input.index) + "_base + j)";
                case Access::fixed:
                    return "_mm256_set1_ps(" + (float16
                        ? "ormi_half_value(" + name + "_fixed)"
                        : name + "_fixed") + ")";
                case Access::indexed:
                    ORMI_THROW(
                        OperationError,
                        "Packed fusion requires planned input access."
                    );
            }

            ORMI_THROW(
                OperationError,
                "Packed fusion received unknown input access."
            );
        }

        std::string packed_binary(
            core::ops::BinaryOp op,
            std::string_view lhs,
            std::string_view rhs
        ) {
            std::string intrinsic;
            switch (op) {
                case core::ops::BinaryOp::add:
                    intrinsic = "_mm256_add_ps";
                    break;
                case core::ops::BinaryOp::sub:
                    intrinsic = "_mm256_sub_ps";
                    break;
                case core::ops::BinaryOp::mul:
                    intrinsic = "_mm256_mul_ps";
                    break;
                case core::ops::BinaryOp::div:
                    intrinsic = "_mm256_div_ps";
                    break;
            }
            return intrinsic + "(" + std::string(lhs) + ", " +
                std::string(rhs) + ")";
        }

        std::size_t math_index(core::ops::UnaryOp op) {
            switch (op) {
                case core::ops::UnaryOp::exp:
                    return 0;
                case core::ops::UnaryOp::log:
                    return 1;
                case core::ops::UnaryOp::sin:
                    return 2;
                case core::ops::UnaryOp::cos:
                    return 3;
                case core::ops::UnaryOp::neg:
                    break;
            }

            ORMI_THROW(
                OperationError,
                "CPU fusion received an invalid packed math operation."
            );
        }
    }

    std::string_view storage_type(core::DType dtype) {
        switch (dtype) {
            case core::DType::float64:
                return "double";
            case core::DType::float32:
                return "float";
            case core::DType::float16:
                return "std::uint16_t";
            case core::DType::int32:
            case core::DType::boolean:
                ORMI_THROW(
                    DTypeError,
                    "CPU fusion code generation requires a floating-point dtype."
                );
        }

        ORMI_THROW(
            DTypeError,
            "CPU fusion code generation received an unknown dtype."
        );
    }

    std::string_view value_type(core::DType dtype) {
        return dtype == core::DType::float64 ? "double" : "float";
    }

    std::string literal(
        core::DType dtype,
        core::runtime::fusion::Constant constant
    ) {
        if (dtype != core::DType::float64) {
            return constant == core::runtime::fusion::Constant::zero
                ? "0.0f"
                : "1.0f";
        }

        return constant == core::runtime::fusion::Constant::zero
            ? "0.0"
            : "1.0";
    }

    std::string value(core::runtime::fusion::Index index) {
        return "v" + std::to_string(index);
    }

    std::string offset(
        std::string_view index,
        const std::vector<core::runtime::fusion::Axis>& axes
    ) {
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
        return result.str();
    }

    std::string output_value(
        core::DType dtype,
        std::string_view expression
    ) {
        if (dtype == core::DType::float16) {
            return "ormi_half_bits(" + std::string(expression) + ")";
        }
        return std::string(expression);
    }

    bool uses_float16(const core::runtime::fusion::Expr& expression) {
        for (const auto& node : expression.nodes) {
            if (node.dtype == core::DType::float16) {
                return true;
            }
        }
        return false;
    }

    void emit_nodes(Code& code, std::string_view indent) {
        for (std::size_t i = 0; i < code.expr.nodes.size(); ++i) {
            const auto& node = code.expr.nodes[i];
            code.out
                << indent << "const " << value_type(node.dtype) << " "
                << value(static_cast<core::runtime::fusion::Index>(i))
                << " = ";
            const auto& instruction = node.instruction;

            if (const auto* input =
                    std::get_if<core::runtime::fusion::Input>(&instruction)) {
                if (input->index >= code.expr.inputs.size()) {
                    ORMI_THROW(
                        OperationError,
                        "Fusion expression references an invalid input."
                    );
                }
                code.out << input_value(node.dtype, *input, code.loop);

            } else if (const auto* constant =
                    std::get_if<core::runtime::fusion::Constant>(
                        &instruction
                    )) {
                code.out << literal(node.dtype, *constant);

            } else if (const auto* cast =
                    std::get_if<core::runtime::fusion::Cast>(&instruction)) {
                require_prior(cast->input, i);
                code.out << cast_value(node.dtype, value(cast->input));

            } else if (const auto* unary_node =
                    std::get_if<core::runtime::fusion::Unary>(&instruction)) {
                require_prior(unary_node->input, i);
                const auto operation = unary(
                    unary_node->op,
                    value(unary_node->input)
                );
                code.out << (
                    is_output(code.expr, i) && !code.expr.reduction
                        ? operation
                        : rounded(node.dtype, operation)
                );

            } else if (const auto* binary_node =
                    std::get_if<core::runtime::fusion::Binary>(&instruction)) {
                require_prior(binary_node->lhs, i);
                require_prior(binary_node->rhs, i);
                const auto operation = value(binary_node->lhs) + " " +
                    std::string(binary(binary_node->op)) + " " +
                    value(binary_node->rhs);
                code.out << (
                    is_output(code.expr, i) && !code.expr.reduction
                        ? operation
                        : rounded(node.dtype, operation)
                );
            }

            code.out << ";\n";
        }
    }

    std::string packed(core::runtime::fusion::Index index) {
        return "p" + std::to_string(index);
    }

    void emit_packed_nodes(Code& code, std::string_view indent) {
        for (std::size_t i = 0; i < code.expr.nodes.size(); ++i) {
            const auto& node = code.expr.nodes[i];
            const auto& instruction = node.instruction;

            if (const auto paired = trig_pair(
                    code.expr,
                    i
                )) {
                if (*paired < i) {
                    continue;
                }

                const auto& unary_node = std::get<
                    core::runtime::fusion::Unary
                >(instruction);
                const auto sine = unary_node.op == core::ops::UnaryOp::sin
                    ? static_cast<core::runtime::fusion::Index>(i)
                    : *paired;
                const auto cosine = unary_node.op == core::ops::UnaryOp::cos
                    ? static_cast<core::runtime::fusion::Index>(i)
                    : *paired;
                code.out
                    << indent << "__m256 " << packed(sine) << ";\n"
                    << indent << "__m256 " << packed(cosine) << ";\n"
                    << indent << "ormi_sincos8(math_ptr, "
                    << packed(unary_node.input) << ", "
                    << packed(sine) << ", " << packed(cosine) << ");\n";
                continue;
            }

            std::string operation;

            if (const auto* input =
                    std::get_if<core::runtime::fusion::Input>(&instruction)) {
                operation = packed_input(*input, node.dtype, code.loop);

            } else if (const auto* constant =
                    std::get_if<core::runtime::fusion::Constant>(
                        &instruction
                    )) {
                operation =
                    *constant == core::runtime::fusion::Constant::zero
                        ? "_mm256_setzero_ps()"
                        : "_mm256_set1_ps(1.0f)";

            } else if (const auto* unary_node =
                    std::get_if<core::runtime::fusion::Unary>(&instruction)) {
                require_prior(unary_node->input, i);
                if (unary_node->op == core::ops::UnaryOp::neg) {
                    operation = "_mm256_sub_ps(_mm256_setzero_ps(), " +
                        packed(unary_node->input) + ")";
                } else if (code.loop.math) {
                    operation =
                        "ormi_math8(math_ptr, " +
                        std::to_string(math_index(unary_node->op)) +
                        ", " + packed(unary_node->input) + ")";
                } else {
                    ORMI_THROW(
                        OperationError,
                        "Packed Float16 fusion only supports arithmetic."
                    );
                }

            } else if (const auto* binary_node =
                    std::get_if<core::runtime::fusion::Binary>(&instruction)) {
                require_prior(binary_node->lhs, i);
                require_prior(binary_node->rhs, i);
                operation = packed_binary(
                    binary_node->op,
                    packed(binary_node->lhs),
                    packed(binary_node->rhs)
                );

            } else if (const auto* cast =
                    std::get_if<core::runtime::fusion::Cast>(&instruction)) {
                require_prior(cast->input, i);
                operation = packed(cast->input);
                if (node.dtype == core::DType::float16) {
                    operation = "ormi_half_round8(" + operation + ")";
                }
            }

            const bool arithmetic =
                std::holds_alternative<core::runtime::fusion::Unary>(
                    instruction
                ) ||
                std::holds_alternative<core::runtime::fusion::Binary>(
                    instruction
                );
            if (
                node.dtype == core::DType::float16 &&
                arithmetic &&
                (!is_output(code.expr, i) || code.expr.reduction)
            ) {
                operation = "ormi_half_round8(" + operation + ")";
            }

            code.out
                << indent << "const __m256 "
                << packed(static_cast<core::runtime::fusion::Index>(i))
                << " = " << operation << ";\n";
        }
    }
}
