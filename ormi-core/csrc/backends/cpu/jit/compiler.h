#pragma once
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "platform/platform.h"

namespace ormi::backends::cpu::jit {
    struct Math;
    struct Target;

    enum class CXX {
        gcc,
        clang,
        clang_cl,
        msvc
    };

    struct Toolchain {
        std::filesystem::path executable;
        CXX id;
        std::vector<std::string> environment;
    };

    std::string_view to_string(CXX id);
    platform::Process run(
        const Toolchain& compiler,
        std::span<const std::string> arguments
    );
    Toolchain resolve();
    const Toolchain& toolchain();
    void build(
        const Toolchain& compiler,
        Target target,
        Math functions,
        const std::filesystem::path& source,
        const std::filesystem::path& output
    );
}
