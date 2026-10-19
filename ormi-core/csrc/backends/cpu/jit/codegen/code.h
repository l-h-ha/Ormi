#pragma once

#include <sstream>
#include <string>
#include <string_view>

#include <ormi/backend/fusion.h>
#include <ormi/core/dtype.h>

#include "backends/cpu/jit/loop.h"

namespace ormi::backends::cpu::jit::codegen {
    struct Code {
        std::ostringstream& out;
        const core::runtime::fusion::Expr& expr;
        const Loop& loop;
    };

    std::string_view storage_type(core::DType dtype);
    std::string_view value_type(core::DType dtype);
    std::string literal(
        core::DType dtype,
        core::runtime::fusion::Constant constant
    );
    std::string value(core::runtime::fusion::Index index);
    std::string output_value(
        core::DType dtype,
        std::string_view expression
    );
    bool uses_float16(const core::runtime::fusion::Expr& expression);

    void emit_nodes(Code& code, std::string_view indent);

    std::string packed(core::runtime::fusion::Index index);
    void emit_packed_nodes(Code& code, std::string_view indent);
}
