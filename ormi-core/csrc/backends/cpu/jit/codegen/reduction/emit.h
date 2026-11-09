#pragma once

#include <string>
#include <string_view>

#include "backends/cpu/jit/codegen/code.h"

namespace ormi::backends::cpu::jit::codegen::reduction {
    inline std::string reduced_value(
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

    inline std::string output_begin(const Code& code) {
        return code.loop.ranged ? "begin" : "0";
    }

    inline std::string output_end(const Code& code) {
        return code.loop.ranged
            ? "end"
            : std::to_string(code.expr.reduction->output.size);
    }

    inline std::string reduced_begin(const Code& code) {
        return code.loop.partial ? "begin" : "0";
    }

    inline std::string reduced_end(const Code& code) {
        return code.loop.partial
            ? "end"
            : std::to_string(code.expr.reduction->reduced.size);
    }

    inline std::string reduction_result(
        const Code& code,
        std::string_view accumulator,
        std::string total = "total"
    ) {
        // Worker partials are combined before mean division and storage conversion.
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

    void emit_scalar(Code& code);
    void emit_packed(Code& code);
    void emit_vectorized(Code& code);
    void emit_scalar_strided(Code& code);
    void emit_vectorized_global(Code& code);
    void emit_packed_output(Code& code);
    void emit_vectorized_output(Code& code);
    void emit_tiled_output(Code& code);
    void emit_tiled_global(Code& code);
}
