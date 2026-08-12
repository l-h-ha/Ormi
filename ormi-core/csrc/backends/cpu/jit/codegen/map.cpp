#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/codegen/map.h"

namespace ormi::backends::cpu::jit::codegen {
    namespace {
        std::string outer_offset(
            const std::vector<core::runtime::fusion::Axis>& axes
        ) {
            std::ostringstream result;
            bool first = true;

            for (const auto& axis : axes) {
                if (axis.divisor == 1) {
                    continue;
                }
                if (!first) {
                    result << " + ";
                }
                first = false;
                result
                    << "((base / " << axis.divisor << ") % "
                    << axis.extent << ")";
                if (axis.stride != 1) {
                    result << " * " << axis.stride;
                }
            }
            return first ? "0" : result.str();
        }

        const core::runtime::fusion::Input& input_node(
            const core::runtime::fusion::Expr& expression,
            std::size_t index
        ) {
            for (const auto& node : expression.nodes) {
                const auto* input =
                    std::get_if<core::runtime::fusion::Input>(
                        &node.instruction
                    );
                if (input && input->index == index) {
                    return *input;
                }
            }

            ORMI_THROW(
                OperationError,
                "Fusion expression is missing input metadata."
            );
        }

        void emit_broadcast_bindings(
            Code& code,
            std::string_view indent
        ) {
            for (std::size_t i = 0; i < code.loop.inputs.size(); ++i) {
                if (code.loop.inputs[i] == Access::span) {
                    code.out
                        << indent << "const auto input" << i << "_base = "
                        << outer_offset(input_node(code.expr, i).axes)
                        << ";\n";
                } else if (code.loop.inputs[i] == Access::fixed) {
                    code.out
                        << indent << "const "
                        << storage_type(code.expr.inputs[i]->dtype())
                        << " input" << i << "_fixed = input" << i << "["
                        << offset(
                            "base",
                            input_node(code.expr, i).axes
                        )
                        << "];\n";
                }
            }
        }

        std::size_t output_count(
            const core::runtime::fusion::Expr& expression
        ) {
            return 1 + expression.auxiliary_outputs.size();
        }

        core::runtime::fusion::Index output_index(
            const core::runtime::fusion::Expr& expression,
            std::size_t output
        ) {
            return output == 0
                ? expression.output
                : expression.auxiliary_outputs[output - 1];
        }

        std::string output_name(std::size_t output) {
            return output == 0
                ? "output"
                : "output" + std::to_string(output);
        }

        void emit_packed_stores(
            Code& code,
            std::string_view indent
        ) {
            for (
                std::size_t output = 0;
                output < output_count(code.expr);
                ++output
            ) {
                const auto index = output_index(code.expr, output);
                const auto name = output_name(output);
                if (code.expr.nodes[index].dtype == core::DType::float16) {
                    code.out
                        << indent << "_mm_storeu_si128(\n"
                        << indent << "    reinterpret_cast<__m128i*>("
                        << name << " + i),\n"
                        << indent << "    ormi_half_bits8("
                        << packed(index) << ")\n"
                        << indent << ");\n";
                } else {
                    code.out
                        << indent << "_mm256_storeu_ps(" << name << " + i, "
                        << packed(index) << ");\n";
                }
            }
        }

        void emit_stores(Code& code, std::string_view indent) {
            for (
                std::size_t output = 0;
                output < output_count(code.expr);
                ++output
            ) {
                const auto index = output_index(code.expr, output);
                code.out
                    << indent << output_name(output) << "[i] = "
                    << output_value(
                        code.expr.nodes[index].dtype,
                        value(index)
                    )
                    << ";\n";
            }
        }

        void emit_packed_map(Code& code) {
            code.out << (code.loop.math
                ? "\n#if defined(ORMI_USE_VECMATH)\n"
                : "\n#if defined(ORMI_USE_F16C)\n");

            if (code.loop.span == 0) {
                code.out
                    << "    unsigned long long i = begin;\n"
                    << "    for (; i + " << code.loop.lanes
                    << " <= end; i += " << code.loop.lanes << ") {\n";
                emit_packed_nodes(code, "        ");
                emit_packed_stores(code, "        ");
                code.out
                    << "    }\n"
                    << "    for (; i < end; ++i) {\n";
                emit_nodes(code, "        ");
                emit_stores(code, "        ");
                code.out << "    }\n";
            } else {
                code.out
                    << "    for (unsigned long long base = begin; "
                    << "base < end; base += " << code.loop.span
                    << ") {\n";
                emit_broadcast_bindings(code, "        ");
                code.out
                    << "        unsigned long long j = 0;\n"
                    << "        for (; j + " << code.loop.lanes << " <= "
                    << code.loop.span << "; j += " << code.loop.lanes
                    << ") {\n"
                    << "            const auto i = base + j;\n";
                emit_packed_nodes(code, "            ");
                emit_packed_stores(code, "            ");
                code.out
                    << "        }\n"
                    << "        for (; j < " << code.loop.span
                    << "; ++j) {\n"
                    << "            const auto i = base + j;\n";
                emit_nodes(code, "            ");
                emit_stores(code, "            ");
                code.out
                    << "        }\n"
                    << "    }\n";
            }

            code.out << "#else\n";
            if (code.loop.span == 0) {
                code.out
                    << "    for (unsigned long long i = begin; "
                    << "i < end; ++i) {\n";
                emit_nodes(code, "        ");
                emit_stores(code, "        ");
                code.out << "    }\n";
            } else {
                code.out
                    << "    for (unsigned long long base = begin; "
                    << "base < end; base += " << code.loop.span
                    << ") {\n";
                emit_broadcast_bindings(code, "        ");
                code.out
                    << "        for (unsigned long long j = 0; j < "
                    << code.loop.span << "; ++j) {\n"
                    << "            const auto i = base + j;\n";
                emit_nodes(code, "            ");
                emit_stores(code, "            ");
                code.out
                    << "        }\n"
                    << "    }\n";
            }
            code.out << "#endif\n";
        }

        void emit_strided_map(Code& code) {
            if (code.loop.tile != 0) {
                code.out
                    << "\n    for (unsigned long long tile0 = 0; tile0 < "
                    << code.expr.shape[0] << "; tile0 += " << code.loop.tile
                    << ") {\n"
                    << "        const auto end0 = tile0 + " << code.loop.tile
                    << " < " << code.expr.shape[0] << " ? tile0 + "
                    << code.loop.tile << " : " << code.expr.shape[0]
                    << ";\n"
                    << "        for (unsigned long long tile1 = 0; tile1 < "
                    << code.expr.shape[1] << "; tile1 += " << code.loop.tile
                    << ") {\n"
                    << "            const auto end1 = tile1 + "
                    << code.loop.tile << " < " << code.expr.shape[1]
                    << " ? tile1 + " << code.loop.tile << " : "
                    << code.expr.shape[1] << ";\n"
                    << "            for (unsigned long long index0 = tile0; "
                    << "index0 < end0; ++index0) {\n"
                    << "                for (unsigned long long index1 = "
                    << "tile1; index1 < end1; ++index1) {\n"
                    << "                    const auto i = index0 * "
                    << code.expr.shape[1] << " + index1;\n";

                emit_nodes(code, "                    ");
                emit_stores(code, "                    ");
                code.out
                    << "                }\n"
                    << "            }\n"
                    << "        }\n"
                    << "    }\n";
                return;
            }

            code.out << "\n    unsigned long long i = 0;\n";
            for (
                std::size_t dimension = 0;
                dimension < code.expr.shape.size();
                ++dimension
            ) {
                const std::string indent((dimension + 1) * 4, ' ');
                code.out
                    << indent << "for (unsigned long long index"
                    << dimension << " = 0; index" << dimension << " < "
                    << code.expr.shape[dimension] << "; ++index"
                    << dimension << ") {\n";
            }

            const std::string body_indent(
                (code.expr.shape.size() + 1) * 4,
                ' '
            );
            emit_nodes(code, body_indent);
            emit_stores(code, body_indent);
            code.out << body_indent << "++i;\n";

            for (
                std::size_t dimension = code.expr.shape.size();
                dimension > 0;
                --dimension
            ) {
                code.out << std::string(dimension * 4, ' ') << "}\n";
            }
        }

        void emit_regular_map(Code& code) {
            const bool nested =
                code.loop.vectorized && code.loop.span != 0;

            if (nested) {
                code.out
                    << "\n    for (unsigned long long base = begin; "
                    << "base < end; base += " << code.loop.span
                    << ") {\n";
                emit_broadcast_bindings(code, "        ");
                code.out
                    << "        ORMI_VECTORIZE_LOOP\n"
                    << "        for (unsigned long long j = 0; j < "
                    << code.loop.span << "; ++j) {\n"
                    << "            const auto i = base + j;\n";
                emit_nodes(code, "            ");
                emit_stores(code, "            ");
                code.out
                    << "        }\n"
                    << "    }\n";
                return;
            }

            code.out << "\n";
            if (code.loop.vectorized) {
                code.out << "    ORMI_VECTORIZE_LOOP\n";
            }
            code.out
                << "    for (unsigned long long i = begin; "
                << "i < end; ++i) {\n";
            emit_nodes(code, "        ");
            emit_stores(code, "        ");
            code.out << "    }\n";
        }
    }

    void emit_map(Code& code) {
        if (code.loop.strided) {
            emit_strided_map(code);
            return;
        }

        if ((uses_float16(code.expr) || code.loop.math) && code.loop.vectorized) {
            emit_packed_map(code);
            return;
        }

        emit_regular_map(code);
    }
}
