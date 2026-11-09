#pragma once
#include <chrono>
#include <optional>
#include <string>
#include <string_view>

namespace ormi::backends::cpu::jit {
    struct Toolchain;

    namespace compatibility {
        inline constexpr std::string_view source_text =
            "#include <bit>\n"
            "#include <cmath>\n"
            "#include <cstddef>\n"
            "#include <cstdint>\n"
            "extern \"C\" int ormi_compiler_probe() {\n"
            "    return static_cast<int>(sizeof(std::size_t));\n"
            "}\n";

        std::optional<std::string> probe(
            const Toolchain& compiler,
            std::chrono::milliseconds timeout
        );
    }
}
