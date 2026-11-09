#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <ormi/backend/fusion.h>

#include "platform/platform.h"

namespace ormi::backends::cpu::jit {
    using Function = void (*)(
        void* const*,
        const void* const*,
        unsigned long long,
        unsigned long long,
        unsigned long long,
        const void*
    );

    struct Compiled {
        std::shared_ptr<platform::Library> library;
        Function function;
        const void* math;
        std::size_t work;
        std::size_t unit;
        std::size_t range;
        bool partial;
    };

    // Load an existing kernel without invoking a compiler.
    bool cached(const core::runtime::fusion::Expr& expression);

    std::shared_ptr<Compiled> compile(
        const core::runtime::fusion::Expr& expression
    );
    std::vector<std::shared_ptr<Compiled>> compile(
        std::span<const core::runtime::fusion::Expr* const> expressions
    );
}
