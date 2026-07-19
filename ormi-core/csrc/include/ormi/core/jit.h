#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace ormi::core::jit {
    struct CacheInfo {
        std::filesystem::path path;
        std::size_t entries;
        std::uintmax_t bytes;
        std::uintmax_t limit;
    };

    CacheInfo info();
    CacheInfo clear();
}
