#pragma once
#include <cstdint>

namespace ormi::core::runtime {
    using BackendId = std::uint16_t;

    inline constexpr BackendId backend_invalid = 0xffff;
}
