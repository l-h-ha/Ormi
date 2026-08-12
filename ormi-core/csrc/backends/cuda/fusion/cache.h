#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace ormi::backends::cuda::fusion::cache {
    struct Artifact {
        std::string ptx;
        bool cached;
    };

    struct Info {
        std::filesystem::path path;
        std::size_t entries;
        std::uintmax_t bytes;
        std::uintmax_t limit;
    };

    using Build = std::function<std::string()>;

    std::filesystem::path directory();
    std::string key(
        std::string_view source,
        std::string_view context
    );
    Artifact get(
        std::string_view source,
        std::string_view context,
        const Build& build
    );
    Info info();
    Info clear();
}
