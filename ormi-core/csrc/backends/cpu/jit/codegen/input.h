#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <ormi/backend/fusion.h>
#include <ormi/core/dtype.h>

#include "backends/cpu/jit/loop.h"

namespace ormi::backends::cpu::jit::codegen {
    std::string offset(
        std::string_view index,
        const std::vector<core::runtime::fusion::Axis>& axes
    );
    std::string input_value(
        core::DType dtype,
        const core::runtime::fusion::Input& input,
        const Loop& loop
    );
    std::string packed_input(
        const core::runtime::fusion::Input& input,
        core::DType dtype,
        const Loop& loop
    );
}
