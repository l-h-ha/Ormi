#pragma once

#include "backends/cpu/jit/codegen/code.h"

namespace ormi::backends::cpu::jit::codegen {
    void emit_reduction(Code& code);
}
