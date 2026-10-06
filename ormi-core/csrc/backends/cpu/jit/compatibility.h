#pragma once
#include <chrono>
#include <optional>
#include <string>

namespace ormi::backends::cpu::jit {
    struct Toolchain;

    namespace compatibility {
        std::optional<std::string> check(
            const Toolchain& compiler,
            std::chrono::milliseconds timeout
        );
        void clear();
    }
}
