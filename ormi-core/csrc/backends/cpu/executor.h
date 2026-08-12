#pragma once

#include <cstddef>

namespace ormi::backends::cpu::executor {
    using Function = void (*)(
        void*,
        std::size_t,
        std::size_t,
        std::size_t
    );

    std::size_t slots(
        std::size_t range,
        std::size_t unit,
        std::size_t work
    );

    void run(
        std::size_t range,
        std::size_t unit,
        std::size_t work,
        Function function,
        void* context
    );
}
