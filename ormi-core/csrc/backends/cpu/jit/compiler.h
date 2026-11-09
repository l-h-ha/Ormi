#pragma once
#include <chrono>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/error.h>

#include "platform/platform.h"

namespace ormi::backends::cpu::jit {
    struct Math;
    struct Target;

    class CompilerUnavailable final : public jitError { using jitError::jitError; };

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
    // Use ORMI_CPU_JIT_CXX when set. Otherwise try the platform's compiler candidates.
    // ORMI_CPU_JIT_CXX_ID selects flag syntax for an override, or filters automatic discovery.
    Toolchain resolve();
    // Cache the first selected compiler. Set compiler environment variables before this call.
    const Toolchain& toolchain();
    // Return false when no compatible compiler is available. Invalid settings throw.
    // Cached kernels can still run without a compiler.
    bool compiler_available();

    std::chrono::milliseconds compiler_timeout();
    platform::Process run(
        const Toolchain& compiler,
        std::span<const std::string> arguments
    );
    void build(
        const Toolchain& compiler,
        Target target,
        Math functions,
        const std::filesystem::path& source,
        const std::filesystem::path& output
    );
}
