#pragma once
#include <cstdint>

namespace ormi::core::runtime {
    using BackendId = std::uint16_t;

    inline constexpr BackendId backend_count = 2;
    inline constexpr BackendId backend_invalid = 0xffff;

    inline constexpr BackendId backend_id_cpu = 0;
    inline constexpr BackendId backend_id_cuda = 1;
}
