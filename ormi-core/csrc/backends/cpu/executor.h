#pragma once

#include <cstddef>

namespace ormi::backends::cpu::executor {
    // Rethrow caller exceptions after workers finish. An exception escaping
    // a worker callback terminates the process. Catch it inside the callback.
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
