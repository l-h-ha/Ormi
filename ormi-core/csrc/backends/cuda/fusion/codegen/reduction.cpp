#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/backend/fusion.h>

#include "backends/cuda/fusion/codegen/emit.h"
#include "backends/cuda/fusion/tile.h"

namespace ormi::backends::cuda::fusion {
    namespace {
        std::string reduced_value(
            core::DType dtype,
            const core::runtime::fusion::Reduction& reduction
        ) {
            if (reduction.op == core::ops::ReduceOp::sum) {
                return "total";
            }

            return "total / static_cast<" + std::string(value_type(dtype)) + ">(" +
                std::to_string(reduction.reduced.size) + "ull)";
        }

        void emit_coordinates(
            std::ostringstream& source,
            const core::runtime::fusion::Expr& expression,
            const std::vector<std::size_t>& dimensions,
            std::string_view flat,
            std::string_view indent
        ) {
            std::size_t stride = 1;
            for (const auto dimension : dimensions) {
                stride *= expression.shape[dimension];
            }

            for (std::size_t i = 0; i < dimensions.size(); ++i) {
                const auto dimension = dimensions[i];
                const auto extent = expression.shape[dimension];
                stride /= extent;
                source
                    << indent << "const unsigned long long index"
                    << dimension << " = ";
                if (extent == 1) {
                    source << "0";
                } else if (dimensions.size() == 1) {
                    source << flat;
                } else if (i == 0) {
                    source << flat << " / " << stride << "ull";
                } else if (stride == 1) {
                    source << flat << " % " << extent << "ull";
                } else {
                    source << "(" << flat << " / " << stride
                        << "ull) % " << extent << "ull";
                }
                source << ";\n";
            }
        }

        std::string tiled_output_offset(
            const core::runtime::fusion::Expr& expression,
            const Tile& tile
        ) {
            auto stride = expression.reduction->output.size;
            std::string result;
            for (const auto dimension : tile.output) {
                stride /= expression.shape[dimension];
                if (!result.empty()) {
                    result += " + ";
                }
                result += "index" + std::to_string(dimension);
                if (stride != 1) {
                    result += " * " + std::to_string(stride);
                }
            }
            return result;
        }
    }

    void emit_reduction(
        std::ostringstream& source,
        const core::runtime::fusion::Expr& expression,
        std::string_view storage,
        std::string_view scalar,
        bool physical
    ) {
        const auto& reduction = *expression.reduction;
        source
            << "    const unsigned long long tasks = "
            << reduction.output.size << "ull * parts;\n"
            << "    for (unsigned long long task = blockIdx.x; task < tasks; task += gridDim.x) {\n"
            << "        const unsigned long long output_index = task / parts;\n"
            << "        const unsigned long long part = task % parts;\n"
            << "        const unsigned long long input_base = "
            << offset("output_index", reduction.output.axes) << ";\n"
            << "        " << scalar << " total = "
            << literal(expression.dtype, core::runtime::fusion::Constant::zero)
            << ";\n\n"
            << "        for (unsigned long long reduced_index =\n"
            << "                 part * blockDim.x + threadIdx.x;\n"
            << "             reduced_index < " << reduction.reduced.size << "ull;\n"
            << "             reduced_index += parts * blockDim.x) {\n"
            << "            const unsigned long long i = input_base + "
            << offset("reduced_index", reduction.reduced.axes) << ";\n";

        emit_nodes(source, expression, nullptr, physical);

        source
            << "            total += " << value(expression.output) << ";\n"
            << "        }\n\n"
            << "        const auto block = ormi_block_sum(total);\n"
            << "        if (threadIdx.x == 0) {\n"
            << "            output[task] = block;\n"
            << "        }\n"
            << "    }\n\n"
            << "}\n\n"
            << "extern \"C\" __global__ void ormi_finish(\n"
            << "    " << storage << "* output,\n"
            << "    const " << scalar << "* partials,\n"
            << "    unsigned long long parts\n"
            << ") {\n"
            << "    for (unsigned long long output_index = blockIdx.x;\n"
            << "         output_index < " << reduction.output.size << "ull;\n"
            << "         output_index += gridDim.x) {\n"
            << "        " << scalar << " total = "
            << literal(expression.dtype, core::runtime::fusion::Constant::zero)
            << ";\n"
            << "        for (unsigned long long part = threadIdx.x; part < parts; part += blockDim.x) {\n"
            << "            total += partials[output_index * parts + part];\n"
            << "        }\n"
            << "        total = ormi_block_sum(total);\n"
            << "        if (threadIdx.x == 0) {\n"
            << "            output[output_index] = "
            << output_value(
                expression.dtype,
                reduced_value(expression.dtype, reduction)
            ) << ";\n"
            << "        }\n"
            << "    }\n"
            << "}\n";
    }

    void emit_tiled_reduction(
        std::ostringstream& source,
        const core::runtime::fusion::Expr& expression,
        const Tile& tile
    ) {
        const auto& reduction = *expression.reduction;
        const auto scalar = value_type(expression.dtype);
        std::vector<std::size_t> outer;
        for (const auto dimension : tile.output) {
            if (dimension != tile.axis) {
                outer.push_back(dimension);
            }
        }
        const auto extent = expression.shape[tile.axis];
        const auto tiles =
            (extent + tile.width - 1) / tile.width;

        source
            << "    (void)parts;\n"
            << "    for (unsigned long long task = blockIdx.x; task < "
            << tile.tasks << "ull; task += gridDim.x) {\n"
            << "        const unsigned long long tile_index = task % "
            << tiles << "ull;\n"
            << "        const unsigned long long outer_index = task / "
            << tiles << "ull;\n";
        emit_coordinates(
            source,
            expression,
            outer,
            "outer_index",
            "        "
        );
        source
            << "        const unsigned long long index" << tile.axis
            << " = tile_index * " << tile.width
            << "ull + threadIdx.x;\n"
            << "        if (index" << tile.axis << " >= " << extent
            << "ull) {\n"
            << "            continue;\n"
            << "        }\n"
            << "        " << scalar << " total = "
            << literal(
                expression.dtype,
                core::runtime::fusion::Constant::zero
            )
            << ";\n"
            << "        for (unsigned long long reduced_index = 0; "
            << "reduced_index < " << reduction.reduced.size
            << "ull; ++reduced_index) {\n";
        emit_coordinates(
            source,
            expression,
            tile.reduced,
            "reduced_index",
            "            "
        );
        emit_nodes(source, expression, &tile);
        source
            << "            total += " << value(expression.output)
            << ";\n"
            << "        }\n"
            << "        output[" << tiled_output_offset(expression, tile)
            << "] = "
            << output_value(
                expression.dtype,
                reduced_value(expression.dtype, reduction)
            )
            << ";\n"
            << "    }\n"
            << "}\n";
    }
}
