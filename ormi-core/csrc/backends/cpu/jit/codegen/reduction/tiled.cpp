#include <cstddef>
#include <string>
#include <vector>

#include "backends/cpu/jit/codegen/reduction/emit.h"

namespace ormi::backends::cpu::jit::codegen::reduction {
    namespace {
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
    }

    void emit_tiled_output(Code& code) {
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

    void emit_tiled_global(Code& code) {
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
}
