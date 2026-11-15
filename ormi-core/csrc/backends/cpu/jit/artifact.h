#pragma once
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "backends/cpu/jit/compile.h"

namespace ormi::backends::cpu::jit::artifact {
    struct Kernel {
        std::size_t work;
        std::size_t math_work;
        std::size_t unit;
        std::size_t range;
        bool partial;
        bool math;
    };

    // Load a cached library without selecting a compiler. Keep its shared cache lock
    // until the shared library has loaded and its kernel symbols have been resolved.
    std::shared_ptr<Compiled> find(
        std::string_view source,
        Kernel kernel
    );
    // Load from disk or compile; rebuild a cached library once if loading fails.
    // Report load failures from fresh builds.
    std::vector<std::shared_ptr<Compiled>> get(
        std::string_view source,
        std::span<const Kernel> kernels
    );
}
