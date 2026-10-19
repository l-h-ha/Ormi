#include <cstddef>
#include <string>
#include <string_view>

#include "backends/cpu/jit/codegen/reduction/emit.h"

namespace ormi::backends::cpu::jit::codegen::reduction {
    namespace {
        std::string open_output_loops(Code& code) {
            code.out
                << "\n    unsigned long long output_index = "
                << output_begin(code) << ";\n";
            for (
                std::size_t level = 0;
                level < code.loop.output_dimensions.size();
                ++level
            ) {
                const auto dimension = code.loop.output_dimensions[level];
                // Worker bounds count output elements; the outer loop counts slabs.
                const auto first = level == 0 && code.loop.ranged;
                const auto begin = first
                    ? (
                        code.loop.unit == 1
                            ? "begin"
                            : "begin / " + std::to_string(code.loop.unit)
                    )
                    : "0";
                const auto end = first
                    ? (
                        code.loop.unit == 1
                            ? "end"
                            : "end / " + std::to_string(code.loop.unit)
                    )
                    : std::to_string(code.expr.shape[dimension]);
                code.out
                    << std::string((level + 1) * 4, ' ')
                    << "for (unsigned long long index" << dimension
                    << " = " << begin << "; index" << dimension << " < "
                    << end << "; ++index" << dimension << ") {\n";
            }

            return std::string((code.loop.output_dimensions.size() + 1) * 4, ' ');
        }

        void close_output_loops(Code& code) {
            for (
                std::size_t level = code.loop.output_dimensions.size();
                level > 0;
                --level
            ) {
                code.out << std::string(level * 4, ' ') << "}\n";
            }
        }
    }

    void emit_vectorized_global(Code& code) {
        const auto scalar = value_type(code.expr.dtype);
        const auto accumulator = code.expr.dtype == core::DType::float64
            ? scalar
            : std::string_view{"double"};
        const auto inner = code.loop.reduced_dimensions.back();
        const auto outer = code.loop.reduced_dimensions.size() - 1;

        code.out << "\n";
        for (const auto dimension : code.loop.output_dimensions) {
            code.out
                << "    const unsigned long long index" << dimension
                << " = 0;\n";
        }
        code.out
            << "    " << accumulator << " total = "
            << literal(
                code.expr.dtype,
                core::runtime::fusion::Constant::zero
            )
            << ";\n";

        for (std::size_t level = 0; level < outer; ++level) {
            const auto dimension =
                code.loop.reduced_dimensions[level];
            const bool split =
                code.loop.split &&
                dimension == *code.loop.split;
            code.out
                << std::string((level + 1) * 4, ' ')
                << "for (unsigned long long index" << dimension
                << " = " << (split ? "begin" : "0") << "; index"
                << dimension << " < "
                << (
                    split
                        ? "end"
                        : std::to_string(code.expr.shape[dimension])
                )
                << "; ++index" << dimension << ") {\n";
        }

        const auto indent = std::string((outer + 1) * 4, ' ');
        const bool split_inner =
            code.loop.split &&
            inner == *code.loop.split;
        const auto inner_begin = split_inner ? "begin" : "0";
        const auto inner_end = split_inner
            ? "end"
            : std::to_string(code.expr.shape[inner]);

        code.out
            << indent << "alignas(64) " << scalar << " partials["
            << code.loop.lanes << "] = {};\n"
            << indent << "unsigned long long inner_index = "
            << inner_begin << ";\n"
            << indent << "for (; inner_index + " << code.loop.lanes
            << " <= " << inner_end << "; inner_index += "
            << code.loop.lanes << ") {\n"
            << indent << "    ORMI_VECTORIZE_LOOP\n"
            << indent << "    for (unsigned long long lane = 0; lane < "
            << code.loop.lanes << "; ++lane) {\n"
            << indent << "        const auto index" << inner
            << " = inner_index + lane;\n";
        emit_nodes(code, indent + "        ");
        code.out
            << indent << "        partials[lane] += "
            << value(code.expr.output) << ";\n"
            << indent << "    }\n"
            << indent << "}\n"
            << indent << "for (unsigned long long lane = 0; lane < "
            << code.loop.lanes << "; ++lane) {\n"
            << indent << "    total += partials[lane];\n"
            << indent << "}\n"
            << indent << "for (; inner_index < " << inner_end
            << "; ++inner_index) {\n"
            << indent << "    const auto index" << inner
            << " = inner_index;\n";
        emit_nodes(code, indent + "    ");
        code.out
            << indent << "    total += "
            << value(code.expr.output) << ";\n"
            << indent << "}\n";

        for (std::size_t level = outer; level > 0; --level) {
            code.out << std::string(level * 4, ' ') << "}\n";
        }
        code.out
            << "    output[0] = "
            << reduction_result(code, accumulator)
            << ";\n";
    }

    void emit_packed_output(Code& code) {
        const auto inner = code.loop.reduced_dimensions.back();
        const auto outer = code.loop.reduced_dimensions.size() - 1;
        constexpr std::size_t block_steps = 32;

        const auto output_indent = open_output_loops(code);
        code.out
            << output_indent << "double total = 0.0;\n"
            << output_indent
            << "__m256 packed_partials = _mm256_setzero_ps();\n"
            << output_indent << "unsigned int block_size = 0;\n";

        for (std::size_t level = 0; level < outer; ++level) {
            const auto dimension =
                code.loop.reduced_dimensions[level];
            code.out
                << std::string(
                    (
                        code.loop.output_dimensions.size() +
                        level +
                        1
                    ) * 4,
                    ' '
                )
                << "for (unsigned long long index" << dimension
                << " = 0; index" << dimension << " < "
                << code.expr.shape[dimension] << "; ++index"
                << dimension << ") {\n";
        }

        const auto vector_indent = std::string(
            (
                code.loop.output_dimensions.size() +
                outer +
                1
            ) * 4,
            ' '
        );
        code.out
            << vector_indent << "unsigned long long inner_index = 0;\n"
            << vector_indent << "for (; inner_index + "
            << code.loop.lanes << " <= " << code.expr.shape[inner]
            << "; inner_index += " << code.loop.lanes << ") {\n";
        emit_packed_nodes(code, vector_indent + "    ");
        code.out
            << vector_indent
            << "    packed_partials = _mm256_add_ps(packed_partials, "
            << packed(code.expr.output) << ");\n"
            << vector_indent << "    if (++block_size == "
            << block_steps << ") {\n"
            << vector_indent << "        alignas(32) float lanes["
            << code.loop.lanes << "];\n"
            << vector_indent
            << "        _mm256_store_ps(lanes, packed_partials);\n"
            << vector_indent
            << "        for (unsigned long long lane = 0; lane < "
            << code.loop.lanes << "; ++lane) {\n"
            << vector_indent << "            total += lanes[lane];\n"
            << vector_indent << "        }\n"
            << vector_indent
            << "        packed_partials = _mm256_setzero_ps();\n"
            << vector_indent << "        block_size = 0;\n"
            << vector_indent << "    }\n"
            << vector_indent << "}\n"
            << vector_indent << "for (; inner_index < "
            << code.expr.shape[inner] << "; ++inner_index) {\n"
            << vector_indent << "    const auto index" << inner
            << " = inner_index;\n";
        emit_nodes(code, vector_indent + "    ");
        code.out
            << vector_indent << "    total += "
            << value(code.expr.output) << ";\n"
            << vector_indent << "}\n";

        for (std::size_t level = outer; level > 0; --level) {
            code.out
                << std::string(
                    (
                        code.loop.output_dimensions.size() +
                        level
                    ) * 4,
                    ' '
                )
                << "}\n";
        }

        code.out
            << output_indent << "alignas(32) float lanes["
            << code.loop.lanes << "];\n"
            << output_indent
            << "_mm256_store_ps(lanes, packed_partials);\n"
            << output_indent
            << "for (unsigned long long lane = 0; lane < "
            << code.loop.lanes << "; ++lane) {\n"
            << output_indent << "    total += lanes[lane];\n"
            << output_indent << "}\n"
            << output_indent << "output[output_index] = "
            << reduction_result(code, "double") << ";\n"
            << output_indent << "++output_index;\n";

        close_output_loops(code);
    }

    void emit_vectorized_output(Code& code) {
        const auto scalar = value_type(code.expr.dtype);
        const auto accumulator = code.expr.dtype == core::DType::float64
            ? scalar
            : std::string_view{"double"};
        const auto inner = code.loop.reduced_dimensions.back();
        const auto outer = code.loop.reduced_dimensions.size() - 1;
        constexpr std::size_t block_steps = 32;

        const auto output_indent = open_output_loops(code);
        code.out
            << output_indent << accumulator << " total = "
            << literal(
                code.expr.dtype,
                core::runtime::fusion::Constant::zero
            )
            << ";\n"
            << output_indent << "alignas(64) " << scalar
            << " partials[" << code.loop.lanes << "] = {};\n"
            << output_indent << "unsigned int block_size = 0;\n";

        for (std::size_t level = 0; level < outer; ++level) {
            const auto dimension =
                code.loop.reduced_dimensions[level];
            code.out
                << std::string(
                    (
                        code.loop.output_dimensions.size() +
                        level +
                        1
                    ) * 4,
                    ' '
                )
                << "for (unsigned long long index" << dimension
                << " = 0; index" << dimension << " < "
                << code.expr.shape[dimension] << "; ++index"
                << dimension << ") {\n";
        }

        const auto vector_indent = std::string(
            (
                code.loop.output_dimensions.size() +
                outer +
                1
            ) * 4,
            ' '
        );
        code.out
            << vector_indent << "unsigned long long inner_index = 0;\n"
            << vector_indent << "for (; inner_index + "
            << code.loop.lanes << " <= " << code.expr.shape[inner]
            << "; inner_index += " << code.loop.lanes << ") {\n"
            << vector_indent << "    ORMI_VECTORIZE_LOOP\n"
            << vector_indent
            << "    for (unsigned long long lane = 0; lane < "
            << code.loop.lanes << "; ++lane) {\n"
            << vector_indent << "        const auto index" << inner
            << " = inner_index + lane;\n";
        emit_nodes(code, vector_indent + "        ");
        code.out
            << vector_indent << "        partials[lane] += "
            << value(code.expr.output) << ";\n"
            << vector_indent << "    }\n"
            << vector_indent << "    if (++block_size == "
            << block_steps << ") {\n"
            << vector_indent
            << "        for (unsigned long long lane = 0; lane < "
            << code.loop.lanes << "; ++lane) {\n"
            << vector_indent << "            total += partials[lane];\n"
            << vector_indent << "            partials[lane] = "
            << literal(
                code.expr.dtype,
                core::runtime::fusion::Constant::zero
            )
            << ";\n"
            << vector_indent << "        }\n"
            << vector_indent << "        block_size = 0;\n"
            << vector_indent << "    }\n"
            << vector_indent << "}\n"
            << vector_indent << "for (; inner_index < "
            << code.expr.shape[inner] << "; ++inner_index) {\n"
            << vector_indent << "    const auto index" << inner
            << " = inner_index;\n";
        emit_nodes(code, vector_indent + "    ");
        code.out
            << vector_indent << "    total += "
            << value(code.expr.output) << ";\n"
            << vector_indent << "}\n";

        for (std::size_t level = outer; level > 0; --level) {
            code.out
                << std::string(
                    (
                        code.loop.output_dimensions.size() +
                        level
                    ) * 4,
                    ' '
                )
                << "}\n";
        }

        code.out
            << output_indent
            << "for (unsigned long long lane = 0; lane < "
            << code.loop.lanes << "; ++lane) {\n"
            << output_indent << "    total += partials[lane];\n"
            << output_indent << "}\n"
            << output_indent << "output[output_index] = "
            << reduction_result(code, accumulator) << ";\n"
            << output_indent << "++output_index;\n";

        close_output_loops(code);
    }

    void emit_scalar_strided(Code& code) {
        const auto& reduction = *code.expr.reduction;
        constexpr std::string_view accumulator = "double";
        const bool blocked = code.expr.dtype != core::DType::float64;
        if (reduction.output.size == 0) {
            return;
        }

        const auto output_indent = open_output_loops(code);
        code.out
            << output_indent << accumulator
            << " total = "
            << literal(
                code.expr.dtype,
                core::runtime::fusion::Constant::zero
            )
            << ";\n";
        if (blocked) {
            code.out
                << output_indent << "float block_total = 0.0f;\n"
                << output_indent << "unsigned int block_size = 0;\n";
        }

        if (reduction.reduced.size != 0) {
            for (
                std::size_t level = 0;
                level < code.loop.reduced_dimensions.size();
                ++level
            ) {
                const auto dimension =
                    code.loop.reduced_dimensions[level];
                const bool split =
                    code.loop.partial &&
                    code.loop.split &&
                    dimension == *code.loop.split;
                const auto begin = split ? "begin" : "0";
                const auto end = split
                    ? "end"
                    : std::to_string(code.expr.shape[dimension]);
                code.out
                    << std::string(
                        (
                            code.loop.output_dimensions.size() +
                            level +
                            1
                        ) * 4,
                        ' '
                    )
                    << "for (unsigned long long index" << dimension
                    << " = " << begin << "; index" << dimension << " < "
                    << end << "; ++index"
                    << dimension << ") {\n";
            }

            const auto reduced_indent = std::string(
                (
                    code.loop.output_dimensions.size() +
                    code.loop.reduced_dimensions.size() +
                    1
                ) * 4,
                ' '
            );
            emit_nodes(code, reduced_indent);
            if (blocked) {
                code.out
                    << reduced_indent << "block_total += "
                    << value(code.expr.output) << ";\n"
                    << reduced_indent << "if (++block_size == 32) {\n"
                    << reduced_indent << "    total += block_total;\n"
                    << reduced_indent << "    block_total = 0.0f;\n"
                    << reduced_indent << "    block_size = 0;\n"
                    << reduced_indent << "}\n";
            } else {
                code.out
                    << reduced_indent << "total += "
                    << value(code.expr.output) << ";\n";
            }

            for (
                std::size_t level =
                    code.loop.reduced_dimensions.size();
                level > 0;
                --level
            ) {
                code.out
                    << std::string(
                        (
                            code.loop.output_dimensions.size() +
                            level
                        ) * 4,
                        ' '
                    )
                    << "}\n";
            }
        }

        if (blocked) {
            code.out << output_indent << "total += block_total;\n";
        }
        code.out
            << output_indent << "output[output_index] = "
            << reduction_result(code, accumulator)
            << ";\n"
            << output_indent << "++output_index;\n";

        close_output_loops(code);
    }

}
