#include <algorithm>

#include "backends/cpu/jit/codegen/reduction.h"
#include "backends/cpu/jit/codegen/reduction/emit.h"

namespace ormi::backends::cpu::jit::codegen {
    namespace {
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
    }

    void emit_reduction(Code& code) {
        if (code.loop.strided) {
            if (code.loop.partial && code.loop.tile != 0) {
                reduction::emit_tiled_global(code);
            } else if (code.loop.partial && code.loop.vectorized) {
                reduction::emit_vectorized_global(code);
            } else if (code.loop.tile != 0) {
                reduction::emit_tiled_output(code);
            } else if (code.loop.vectorized) {
                if (can_pack_output(code)) {
                    code.out << "\n#if defined(ORMI_USE_AVX2)\n";
                    reduction::emit_packed_output(code);
                    code.out << "#else\n";
                    reduction::emit_vectorized_output(code);
                    code.out << "#endif\n";
                } else {
                    reduction::emit_vectorized_output(code);
                }
            } else {
                reduction::emit_scalar_strided(code);
            }
            return;
        }

        if (uses_float16(code.expr) && code.loop.vectorized) {
            reduction::emit_packed(code);
        } else if (code.loop.vectorized) {
            reduction::emit_vectorized(code);
        } else {
            code.out << "\n";
            reduction::emit_scalar(code);
        }
    }
}
