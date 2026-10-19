#pragma once

#include <sstream>

namespace ormi::backends::cpu::jit::codegen {
    void emit_headers(
        std::ostringstream& source,
        bool float16,
        bool transcendental,
        bool intrinsics,
        bool vectorized
    );
    void emit_float16_support(std::ostringstream& source);
    void emit_math_support(std::ostringstream& source);
}
