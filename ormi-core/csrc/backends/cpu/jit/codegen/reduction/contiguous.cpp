#include <cstddef>
#include <string>
#include <string_view>

#include "backends/cpu/jit/codegen/input.h"
#include "backends/cpu/jit/codegen/reduction/emit.h"

namespace ormi::backends::cpu::jit::codegen::reduction {
    void emit_scalar(Code& code) {
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

    void emit_packed(Code& code) {
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
        emit_scalar(code);
        code.out << "#endif\n";
    }

    void emit_vectorized(Code& code) {
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
