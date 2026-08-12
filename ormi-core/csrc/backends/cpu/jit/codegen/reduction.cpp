#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/ops/kind.h>

#include "backends/cpu/jit/codegen/reduction.h"

namespace ormi::backends::cpu::jit::codegen {
    namespace {
        std::string reduced_value(
            const core::runtime::fusion::Reduction& reduction,
            std::string_view accumulator,
            std::string total = "total"
        ) {
            if (reduction.op == core::ops::ReduceOp::sum) {
                return total;
            }

            return total + " / static_cast<" +
                std::string(accumulator) + ">(" +
                std::to_string(reduction.reduced.size) + "ull)";
        }

        std::string output_begin(const Code& code) {
            return code.loop.ranged ? "begin" : "0";
        }

        std::string output_end(const Code& code) {
            return code.loop.ranged
                ? "end"
                : std::to_string(code.expr.reduction->output.size);
        }

        std::string reduced_begin(const Code& code) {
            return code.loop.partial ? "begin" : "0";
        }

        std::string reduced_end(const Code& code) {
            return code.loop.partial
                ? "end"
                : std::to_string(code.expr.reduction->reduced.size);
        }

        std::string reduction_result(
            const Code& code,
            std::string_view accumulator,
            std::string total = "total"
        ) {
            if (code.loop.partial) {
                return total;
            }
            return output_value(
                code.expr.dtype,
                reduced_value(
                    *code.expr.reduction,
                    accumulator,
                    total
                )
            );
        }

        std::string tiled_output_index(
            const Code& code,
            std::size_t tile_axis
        ) {
            std::string result;
            auto stride = code.expr.reduction->output.size;
            for (const auto dimension : code.loop.output_dimensions) {
                stride /= code.expr.shape[dimension];
                if (!result.empty()) {
                    result += " + ";
                }
                const auto tiled = dimension == tile_axis;
                if (tiled && stride != 1) {
                    result += "(tile + lane)";
                } else {
                    result += tiled
                        ? "tile + lane"
                        : "index" + std::to_string(dimension);
                }
                if (stride != 1) {
                    result += " * " + std::to_string(stride);
                }
            }
            return result;
        }

        void emit_tiled_output_reduction(Code& code) {
            const auto& reduction = *code.expr.reduction;
            const auto tile_axis = *code.loop.tile_axis;
            const auto slab =
                reduction.output.size / code.expr.shape[tile_axis];
            const auto tile_begin = code.loop.ranged
                ? (
                    slab == 1
                        ? "begin"
                        : "begin / " + std::to_string(slab)
                )
                : "0";
            const auto tile_end = code.loop.ranged
                ? (
                    slab == 1
                        ? "end"
                        : "end / " + std::to_string(slab)
                )
                : std::to_string(code.expr.shape[tile_axis]);

            code.out
                << "\n    for (unsigned long long tile = "
                << tile_begin << "; tile < " << tile_end << "; tile += "
                << code.loop.tile << ") {\n"
                << "        const auto width = tile + " << code.loop.tile
                << " < " << tile_end << " ? " << code.loop.tile
                << " : " << tile_end << " - tile;\n";

            std::size_t output_level = 0;
            for (const auto dimension : code.loop.output_dimensions) {
                if (dimension == tile_axis) {
                    continue;
                }
                code.out
                    << std::string((output_level + 2) * 4, ' ')
                    << "for (unsigned long long index" << dimension
                    << " = 0; index" << dimension << " < "
                    << code.expr.shape[dimension] << "; ++index"
                    << dimension << ") {\n";
                ++output_level;
            }

            const auto output_indent =
                std::string((output_level + 2) * 4, ' ');
            code.out
                << output_indent << "alignas(64) "
                << value_type(code.expr.dtype) << " totals["
                << code.loop.tile << "] = {};\n";

            for (
                std::size_t level = 0;
                level < code.loop.reduced_dimensions.size();
                ++level
            ) {
                const auto dimension =
                    code.loop.reduced_dimensions[level];
                code.out
                    << std::string(
                        (output_level + level + 2) * 4,
                        ' '
                    )
                    << "for (unsigned long long index" << dimension
                    << " = 0; index" << dimension << " < "
                    << code.expr.shape[dimension] << "; ++index"
                    << dimension << ") {\n";
            }

            const auto reduced_indent = std::string(
                (
                    output_level +
                    code.loop.reduced_dimensions.size() +
                    2
                ) * 4,
                ' '
            );
            code.out
                << reduced_indent << "ORMI_VECTORIZE_LOOP\n"
                << reduced_indent
                << "for (unsigned long long lane = 0; "
                << "lane < width; ++lane) {\n"
                << reduced_indent << "    const auto index" << tile_axis
                << " = tile + lane;\n";
            emit_nodes(code, reduced_indent + "    ");
            code.out
                << reduced_indent << "    totals[lane] += "
                << value(code.expr.output) << ";\n"
                << reduced_indent << "}\n";

            for (
                std::size_t level =
                    code.loop.reduced_dimensions.size();
                level > 0;
                --level
            ) {
                code.out
                    << std::string(
                        (output_level + level + 1) * 4,
                        ' '
                    )
                    << "}\n";
            }

            code.out
                << output_indent << "ORMI_VECTORIZE_LOOP\n"
                << output_indent
                << "for (unsigned long long lane = 0; "
                << "lane < width; ++lane) {\n"
                << output_indent << "    output["
                << tiled_output_index(code, tile_axis) << "] = "
                << output_value(
                    code.expr.dtype,
                    reduced_value(
                        reduction,
                        value_type(code.expr.dtype),
                        "totals[lane]"
                    )
                )
                << ";\n"
                << output_indent << "}\n";

            for (std::size_t level = output_level; level > 0; --level) {
                code.out << std::string((level + 1) * 4, ' ') << "}\n";
            }
            code.out << "    }\n";
        }

        void emit_tiled_global_reduction(Code& code) {
            const auto outer = *code.loop.split;
            const auto inner = *code.loop.tile_axis;
            const auto scalar = value_type(code.expr.dtype);
            std::vector<std::size_t> fixed;
            for (const auto dimension : code.loop.reduced_dimensions) {
                if (dimension != outer && dimension != inner) {
                    fixed.push_back(dimension);
                }
            }

            code.out
                << "\n    " << scalar << " total = "
                << literal(
                    code.expr.dtype,
                    core::runtime::fusion::Constant::zero
                )
                << ";\n";

            for (std::size_t level = 0; level < fixed.size(); ++level) {
                const auto dimension = fixed[level];
                code.out
                    << std::string((level + 1) * 4, ' ')
                    << "for (unsigned long long index" << dimension
                    << " = 0; index" << dimension << " < "
                    << code.expr.shape[dimension] << "; ++index"
                    << dimension << ") {\n";
            }

            const auto indent = std::string((fixed.size() + 1) * 4, ' ');
            code.out
                << indent << "for (unsigned long long tile" << outer
                << " = begin; tile" << outer << " < end; tile" << outer
                << " += " << code.loop.tile << ") {\n"
                << indent << "    const auto end" << outer << " = tile"
                << outer
                << " + " << code.loop.tile << " < end ? tile" << outer
                << " + " << code.loop.tile << " : end;\n"
                << indent << "    for (unsigned long long tile" << inner
                << " = 0; tile" << inner << " < "
                << code.expr.shape[inner] << "; tile" << inner << " += "
                << code.loop.tile << ") {\n"
                << indent << "        const auto end" << inner << " = tile"
                << inner << " + " << code.loop.tile << " < "
                << code.expr.shape[inner] << " ? tile" << inner << " + "
                << code.loop.tile << " : " << code.expr.shape[inner]
                << ";\n"
                << indent << "        for (unsigned long long index" << outer
                << " = tile" << outer << "; index" << outer << " < end"
                << outer << "; ++index" << outer << ") {\n"
                << indent
                << "            for (unsigned long long index" << inner
                << " = tile" << inner << "; index" << inner << " < end"
                << inner << "; ++index" << inner << ") {\n";
            emit_nodes(code, indent + "                ");
            code.out
                << indent << "                total += "
                << value(code.expr.output) << ";\n"
                << indent << "            }\n"
                << indent << "        }\n"
                << indent << "    }\n"
                << indent << "}\n";

            for (std::size_t level = fixed.size(); level > 0; --level) {
                code.out << std::string(level * 4, ' ') << "}\n";
            }
            code.out
                << "    output[0] = "
                << reduction_result(code, scalar) << ";\n";
        }

        void emit_vectorized_strided_reduction(Code& code) {
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

        bool can_pack_output(const Code& code) {
            return
                code.expr.dtype == core::DType::float32 &&
                code.loop.lanes == 8 &&
                !code.loop.reduced_dimensions.empty() &&
                std::all_of(
                    code.expr.nodes.begin(),
                    code.expr.nodes.end(),
                    [](const core::runtime::fusion::Node& node) {
                        return node.dtype == core::DType::float32;
                    }
                );
        }

        void emit_packed_output_reduction(Code& code) {
            const auto inner = code.loop.reduced_dimensions.back();
            const auto outer = code.loop.reduced_dimensions.size() - 1;
            constexpr std::size_t block_steps = 32;

            code.out
                << "\n    unsigned long long output_index = "
                << output_begin(code) << ";\n";
            for (
                std::size_t level = 0;
                level < code.loop.output_dimensions.size();
                ++level
            ) {
                const auto dimension =
                    code.loop.output_dimensions[level];
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

            const auto output_indent = std::string(
                (code.loop.output_dimensions.size() + 1) * 4,
                ' '
            );
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

            for (
                std::size_t level = code.loop.output_dimensions.size();
                level > 0;
                --level
            ) {
                code.out << std::string(level * 4, ' ') << "}\n";
            }
        }

        void emit_vectorized_output_reduction(Code& code) {
            const auto scalar = value_type(code.expr.dtype);
            const auto accumulator = code.expr.dtype == core::DType::float64
                ? scalar
                : std::string_view{"double"};
            const auto inner = code.loop.reduced_dimensions.back();
            const auto outer = code.loop.reduced_dimensions.size() - 1;
            constexpr std::size_t block_steps = 32;

            code.out
                << "\n    unsigned long long output_index = "
                << output_begin(code) << ";\n";
            for (
                std::size_t level = 0;
                level < code.loop.output_dimensions.size();
                ++level
            ) {
                const auto dimension =
                    code.loop.output_dimensions[level];
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

            const auto output_indent = std::string(
                (code.loop.output_dimensions.size() + 1) * 4,
                ' '
            );
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

            for (
                std::size_t level = code.loop.output_dimensions.size();
                level > 0;
                --level
            ) {
                code.out << std::string(level * 4, ' ') << "}\n";
            }
        }

        void emit_strided_reduction(Code& code) {
            if (code.loop.partial && code.loop.tile != 0) {
                emit_tiled_global_reduction(code);
                return;
            }
            if (code.loop.partial && code.loop.vectorized) {
                emit_vectorized_strided_reduction(code);
                return;
            }
            if (code.loop.tile != 0) {
                emit_tiled_output_reduction(code);
                return;
            }
            if (code.loop.vectorized) {
                if (can_pack_output(code)) {
                    code.out << "\n#if defined(ORMI_USE_AVX2)\n";
                    emit_packed_output_reduction(code);
                    code.out << "#else\n";
                    emit_vectorized_output_reduction(code);
                    code.out << "#endif\n";
                } else {
                    emit_vectorized_output_reduction(code);
                }
                return;
            }

            const auto& reduction = *code.expr.reduction;
            constexpr std::string_view accumulator = "double";
            const bool blocked = code.expr.dtype != core::DType::float64;
            if (reduction.output.size == 0) {
                return;
            }

            code.out
                << "\n    unsigned long long output_index = "
                << output_begin(code) << ";\n";
            for (
                std::size_t level = 0;
                level < code.loop.output_dimensions.size();
                ++level
            ) {
                const auto dimension =
                    code.loop.output_dimensions[level];
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

            const auto output_indent = std::string(
                (code.loop.output_dimensions.size() + 1) * 4,
                ' '
            );
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

            for (
                std::size_t level = code.loop.output_dimensions.size();
                level > 0;
                --level
            ) {
                code.out << std::string(level * 4, ' ') << "}\n";
            }
        }

        void emit_scalar_reduction(Code& code) {
            const auto& reduction = *code.expr.reduction;
            constexpr std::string_view accumulator = "double";
            const bool blocked = code.expr.dtype != core::DType::float64;
            code.out
                << "    for (unsigned long long output_index = "
                << output_begin(code) << "; output_index < "
                << output_end(code)
                << "; ++output_index) {\n"
                << "        " << accumulator << " total = "
                << literal(
                    code.expr.dtype,
                    core::runtime::fusion::Constant::zero
                )
                << ";\n";
            if (blocked) {
                code.out
                    << "        float block_total = 0.0f;\n"
                    << "        unsigned int block_size = 0;\n";
            }
            code.out
                << "        const auto input_base = "
                << offset("output_index", reduction.output.axes) << ";\n"
                << "        for (unsigned long long reduced_index = "
                << reduced_begin(code) << "; reduced_index < "
                << reduced_end(code)
                << "; ++reduced_index) {\n"
                << "            const auto i = input_base + "
                << offset("reduced_index", reduction.reduced.axes)
                << ";\n";
            emit_nodes(code, "            ");
            if (blocked) {
                code.out
                    << "            block_total += "
                    << value(code.expr.output) << ";\n"
                    << "            if (++block_size == 32) {\n"
                    << "                total += block_total;\n"
                    << "                block_total = 0.0f;\n"
                    << "                block_size = 0;\n"
                    << "            }\n";
            } else {
                code.out
                    << "            total += " << value(code.expr.output)
                    << ";\n";
            }
            code.out << "        }\n";
            if (blocked) {
                code.out << "        total += block_total;\n";
            }
            code.out
                << "        output[output_index] = "
                << reduction_result(code, accumulator)
                << ";\n"
                << "    }\n";
        }

        void emit_packed_reduction(Code& code) {
            const auto& reduction = *code.expr.reduction;
            constexpr std::size_t block_steps = 32;
            const auto block_elements = code.loop.lanes * block_steps;
            code.out
                << "\n#if defined(ORMI_USE_F16C)\n"
                << "    for (unsigned long long output_index = "
                << output_begin(code) << "; output_index < "
                << output_end(code)
                << "; ++output_index) {\n"
                << "        double total = 0.0;\n"
                << "        const auto input_base = "
                << offset("output_index", reduction.output.axes) << ";\n"
                << "        unsigned long long reduced_index = "
                << reduced_begin(code) << ";\n"
                << "        for (; reduced_index + " << block_elements
                << " <= " << reduced_end(code)
                << "; reduced_index += " << block_elements << ") {\n"
                << "            __m256 partials = _mm256_setzero_ps();\n"
                << "            for (unsigned int block = 0; block < "
                << block_steps << "; ++block) {\n"
                << "                const auto i = input_base + "
                << "reduced_index + block * " << code.loop.lanes << ";\n";
            emit_packed_nodes(code, "                ");
            code.out
                << "                partials = _mm256_add_ps(partials, "
                << packed(code.expr.output) << ");\n"
                << "            }\n"
                << "            alignas(32) float lanes[" << code.loop.lanes
                << "];\n"
                << "            _mm256_store_ps(lanes, partials);\n"
                << "            for (unsigned long long lane = 0; lane < "
                << code.loop.lanes << "; ++lane) {\n"
                << "                total += lanes[lane];\n"
                << "            }\n"
                << "        }\n"
                << "        __m256 partials = _mm256_setzero_ps();\n"
                << "        for (; reduced_index + " << code.loop.lanes
                << " <= " << reduced_end(code)
                << "; reduced_index += " << code.loop.lanes << ") {\n"
                << "            const auto i = input_base + "
                << "reduced_index;\n";
            emit_packed_nodes(code, "            ");
            code.out
                << "            partials = _mm256_add_ps(partials, "
                << packed(code.expr.output) << ");\n"
                << "        }\n"
                << "        alignas(32) float lanes[" << code.loop.lanes
                << "];\n"
                << "        _mm256_store_ps(lanes, partials);\n"
                << "        for (unsigned long long lane = 0; lane < "
                << code.loop.lanes << "; ++lane) {\n"
                << "            total += lanes[lane];\n"
                << "        }\n"
                << "        for (; reduced_index < "
                << reduced_end(code) << "; ++reduced_index) {\n"
                << "            const auto i = input_base + "
                << "reduced_index;\n";
            emit_nodes(code, "            ");
            code.out
                << "            total += " << value(code.expr.output)
                << ";\n"
                << "        }\n"
                << "        output[output_index] = "
                << reduction_result(code, "double")
                << ";\n"
                << "    }\n"
                << "#else\n";
            emit_scalar_reduction(code);
            code.out << "#endif\n";
        }

        void emit_vectorized_reduction(Code& code) {
            const auto& reduction = *code.expr.reduction;
            const auto scalar = value_type(code.expr.dtype);
            const bool blocked = code.expr.dtype != core::DType::float64;
            const auto accumulator = blocked
                ? std::string_view{"double"}
                : scalar;
            constexpr std::size_t block_steps = 32;
            code.out
                << "\n    const auto ormi_map = "
                << "[&](unsigned long long i) noexcept -> "
                << scalar << " {\n";
            emit_nodes(code, "        ");
            code.out
                << "        return " << value(code.expr.output) << ";\n"
                << "    };\n\n"
                << "    for (unsigned long long output_index = "
                << output_begin(code) << "; output_index < "
                << output_end(code)
                << "; ++output_index) {\n"
                << "        " << accumulator << " total = "
                << literal(
                    code.expr.dtype,
                    core::runtime::fusion::Constant::zero
                )
                << ";\n"
                << "        const auto input_base = "
                << offset("output_index", reduction.output.axes) << ";\n"
                << "        unsigned long long reduced_index = "
                << reduced_begin(code) << ";\n";
            if (blocked) {
                const auto block_elements = code.loop.lanes * block_steps;
                code.out
                    << "        for (; reduced_index + " << block_elements
                    << " <= " << reduced_end(code)
                    << "; reduced_index += " << block_elements << ") {\n"
                    << "            alignas(64) float partials["
                    << code.loop.lanes << "] = {};\n"
                    << "            for (unsigned int block = 0; block < "
                    << block_steps << "; ++block) {\n"
                    << "                ORMI_VECTORIZE_LOOP\n"
                    << "                for (unsigned long long lane = 0; "
                    << "lane < " << code.loop.lanes << "; ++lane) {\n"
                    << "                    partials[lane] += ormi_map("
                    << "input_base + reduced_index + block * "
                    << code.loop.lanes << " + lane);\n"
                    << "                }\n"
                    << "            }\n"
                    << "            for (unsigned long long lane = 0; lane < "
                    << code.loop.lanes << "; ++lane) {\n"
                    << "                total += partials[lane];\n"
                    << "            }\n"
                    << "        }\n";
            }
            code.out
                << "        alignas(64) " << scalar << " partials["
                << code.loop.lanes << "] = {};\n"
                << "        for (; reduced_index + " << code.loop.lanes
                << " <= " << reduced_end(code)
                << "; reduced_index += " << code.loop.lanes << ") {\n"
                << "            ORMI_VECTORIZE_LOOP\n"
                << "            for (unsigned long long lane = 0; lane < "
                << code.loop.lanes << "; ++lane) {\n"
                << "                partials[lane] += ormi_map("
                << "input_base + reduced_index + lane);\n"
                << "            }\n"
                << "        }\n"
                << "        for (unsigned long long lane = 0; lane < "
                << code.loop.lanes << "; ++lane) {\n"
                << "            total += partials[lane];\n"
                << "        }\n"
                << "        for (; reduced_index < "
                << reduced_end(code) << "; ++reduced_index) {\n"
                << "            total += ormi_map(input_base + "
                << "reduced_index);\n"
                << "        }\n"
                << "        output[output_index] = "
                << reduction_result(code, accumulator)
                << ";\n"
                << "    }\n";
        }
    }

    void emit_reduction(Code& code) {
        if (code.loop.strided) {
            emit_strided_reduction(code);
            return;
        }

        if (uses_float16(code.expr) && code.loop.vectorized) {
            emit_packed_reduction(code);
            return;
        }

        if (code.loop.vectorized) {
            emit_vectorized_reduction(code);
            return;
        }

        code.out << "\n";
        emit_scalar_reduction(code);
    }
}
