#pragma once
#include <string>
#include <string_view>
#include <vector>

#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/math.h"

namespace ormi::backends::cpu::jit {
    struct Target {
        bool avx2 = false;
        bool f16c = false;

        bool operator==(const Target&) const = default;
    };

    Target select_target();
    std::string to_string(Target target);
    std::vector<std::string> target_flags(CXX compiler, Target target);

    const Target& selected_target();
    Math selected_math(CXX compiler) noexcept;
    std::string context(Math functions);
    std::string_view target_context();
}
