#include <algorithm>
#include <cstddef>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/codegen.h"
#include "backends/cpu/jit/codegen/code.h"
#include "backends/cpu/jit/codegen/map.h"
#include "backends/cpu/jit/codegen/reduction.h"
#include "backends/cpu/jit/codegen/support.h"
#include "backends/cpu/jit/loop.h"

namespace ormi::backends::cpu::jit {
    namespace {
        bool uses_transcendentals(
            const core::runtime::fusion::Expr& expression
        ) noexcept {
            return std::any_of(
                expression.nodes.begin(),
                expression.nodes.end(),
                [](const core::runtime::fusion::Node& node) {
                    const auto* unary = std::get_if<
                        core::runtime::fusion::Unary
                    >(&node.instruction);
                    return unary && unary->op != core::ops::UnaryOp::neg;
                }
            );
        }

        bool uses_intrinsics(
            const core::runtime::fusion::Expr& expression,
            const Loop& loop,
            bool float16
        ) noexcept {
            return float16 || loop.math ||
                (loop.vectorized && expression.reduction.has_value());
        }

        void emit_signature(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression,
            const Loop& loop,
            std::string_view symbol
        ) {
            const auto output_type = loop.partial
                ? codegen::value_type(expression.dtype)
                : codegen::storage_type(expression.dtype);
            source
                << "extern \"C\" ORMI_EXPORT void " << symbol << "(\n"
                << "    void* const* output_ptrs,\n"
                << "    const void* const* input_ptrs,\n"
                << "    unsigned long long elements,\n"
                << "    unsigned long long begin,\n"
                << "    unsigned long long end,\n"
                << "    const void* math_ptr\n"
                << ") {\n";

            if (loop.vectorized && !loop.partial) {
                source
                    << "    if (elements == 0) {\n"
                    << "        return;\n"
                    << "    }\n\n"
                    << "    auto* ORMI_RESTRICT output = "
                    << "ormi_assume_aligned(\n"
                    << "        static_cast<"
                    << output_type
                    << "*>(output_ptrs[0])\n"
                    << "    );\n";
            } else {
                source
                    << "    auto* output = static_cast<"
                    << output_type
                    << "*>(output_ptrs[0]);\n";
            }

            for (
                std::size_t i = 0;
                i < expression.auxiliary_outputs.size();
                ++i
            ) {
                const auto output = expression.auxiliary_outputs[i];
                const auto type = codegen::storage_type(
                    expression.nodes[output].dtype
                );
                if (loop.vectorized && !loop.partial) {
                    source
                        << "    auto* ORMI_RESTRICT output" << i + 1
                        << " = ormi_assume_aligned(\n"
                        << "        static_cast<" << type
                        << "*>(output_ptrs[" << i + 1 << "])\n"
                        << "    );\n";
                } else {
                    source
                        << "    auto* output" << i + 1
                        << " = static_cast<" << type
                        << "*>(output_ptrs[" << i + 1 << "]);\n";
                }
            }

            for (
                std::size_t i = 0;
                i < expression.inputs.size();
                ++i
            ) {
                if (loop.vectorized) {
                    source
                        << "    const auto* input" << i
                        << " = ormi_assume_aligned(\n"
                        << "        static_cast<const "
                        << codegen::storage_type(
                            expression.inputs[i]->dtype()
                        )
                        << "*>(input_ptrs[" << i << "])\n"
                        << "    );\n";
                } else {
                    source
                        << "    const auto* input" << i
                        << " = static_cast<const "
                        << codegen::storage_type(
                            expression.inputs[i]->dtype()
                        )
                        << "*>(input_ptrs[" << i << "]);\n";
                }
            }

            if (loop.vectorized) {
                for (std::size_t i = 0; i < loop.inputs.size(); ++i) {
                    if (loop.inputs[i] == Access::scalar) {
                        source
                            << "    const "
                            << codegen::storage_type(
                                expression.inputs[i]->dtype()
                            )
                            << " input" << i << "_scalar = input" << i
                            << "[0];\n";
                    }
                }
            }
        }

        void validate(const core::runtime::fusion::Expr& expression) {
            if (
                expression.nodes.empty() ||
                expression.output >= expression.nodes.size() ||
                std::any_of(
                    expression.auxiliary_outputs.begin(),
                    expression.auxiliary_outputs.end(),
                    [&expression](const auto output) {
                        return output >= expression.nodes.size();
                    }
                )
            ) {
                ORMI_THROW(
                    OperationError,
                    "Cannot generate code for an empty or malformed fusion expression."
                );
            }

            if (expression.reduction && !expression.auxiliary_outputs.empty()) {
                ORMI_THROW(
                    OperationError,
                    "CPU fusion does not support auxiliary reduction outputs."
                );
            }

            if (
                expression.reduction &&
                expression.reduction->op != core::ops::ReduceOp::sum &&
                expression.reduction->op != core::ops::ReduceOp::mean
            ) {
                ORMI_THROW(
                    OperationError,
                    "CPU fusion received an unsupported reduction."
                );
            }
        }

        void emit_kernel(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression,
            const Loop& loop,
            std::string_view symbol
        ) {
            emit_signature(source, expression, loop, symbol);

            codegen::Code code{source, expression, loop};
            if (expression.reduction) {
                codegen::emit_reduction(code);
            } else {
                codegen::emit_map(code);
            }
            source << "}\n";
        }
    }

    std::string emit(
        const core::runtime::fusion::Expr& expression,
        const Loop& loop
    ) {
        validate(expression);

        const bool float16 = codegen::uses_float16(expression);
        std::ostringstream source;
        codegen::emit_headers(
            source,
            float16,
            uses_transcendentals(expression),
            uses_intrinsics(expression, loop, float16),
            loop.vectorized
        );
        if (float16) {
            codegen::emit_float16_support(source);
        }
        if (loop.math) {
            codegen::emit_math_support(source);
        }
        emit_kernel(source, expression, loop, "ormi_kernel");
        return source.str();
    }

    std::string emit(
        std::span<const core::runtime::fusion::Expr* const> expressions,
        std::span<const Loop> loops
    ) {
        if (expressions.empty() || expressions.size() != loops.size()) {
            ORMI_THROW(
                OperationError,
                "CPU JIT batch requires one loop plan per expression."
            );
        }

        bool float16 = false;
        bool transcendental = false;
        bool intrinsics = false;
        bool math = false;
        bool vectorized = false;
        for (std::size_t i = 0; i < expressions.size(); ++i) {
            if (!expressions[i]) {
                ORMI_THROW(OperationError, "CPU JIT batch contains a null expression.");
            }
            validate(*expressions[i]);
            const bool expression_float16 =
                codegen::uses_float16(*expressions[i]);
            float16 = float16 || expression_float16;
            transcendental = transcendental ||
                uses_transcendentals(*expressions[i]);
            intrinsics = intrinsics || uses_intrinsics(
                *expressions[i],
                loops[i],
                expression_float16
            );
            math = math || loops[i].math;
            vectorized = vectorized || loops[i].vectorized;
        }

        std::ostringstream source;
        codegen::emit_headers(
            source,
            float16,
            transcendental,
            intrinsics,
            vectorized
        );
        if (float16) {
            codegen::emit_float16_support(source);
        }
        if (math) {
            codegen::emit_math_support(source);
        }
        for (std::size_t i = 0; i < expressions.size(); ++i) {
            emit_kernel(
                source,
                *expressions[i],
                loops[i],
                "ormi_kernel_" + std::to_string(i)
            );
        }
        return source.str();
    }

    std::string emit(const core::runtime::fusion::Expr& expression) {
        return emit(expression, plan(expression));
    }
}
