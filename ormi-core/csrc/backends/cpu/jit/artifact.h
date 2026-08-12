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

    std::shared_ptr<Compiled> find(
        std::string_view source,
        Kernel kernel
    );
    std::vector<std::shared_ptr<Compiled>> get(
        std::string_view source,
        std::span<const Kernel> kernels
    );
}
