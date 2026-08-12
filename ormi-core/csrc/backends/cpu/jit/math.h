#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include <ormi/backend/fusion.h>

namespace ormi::backends::cpu::jit {
    using Unary8 = void (*)(float* output, const float* input);

    enum class MathABI {
        buffered,
        direct
    };

    struct Math {
        const void* functions = nullptr;
        MathABI abi = MathABI::buffered;
    };

    std::optional<core::runtime::fusion::Index> trig_pair(
        const core::runtime::fusion::Expr& expression,
        std::size_t index
    ) noexcept;

    Math math(MathABI abi) noexcept;
}
