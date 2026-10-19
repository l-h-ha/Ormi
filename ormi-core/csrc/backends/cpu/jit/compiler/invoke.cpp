#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/math.h"
#include "backends/cpu/jit/target.h"
#include "platform/platform.h"

namespace ormi::backends::cpu::jit {
    namespace {
        constexpr std::uintmax_t default_timeout_ms = 120000;

        std::vector<std::string> command(
            const Toolchain& compiler,
            Target target,
            Math functions,
            const std::filesystem::path& source,
            const std::filesystem::path& output
        ) {
            const bool vector_math = functions.functions != nullptr;
            const bool direct_math =
                vector_math && functions.abi == MathABI::direct;

            if (compiler.id == CXX::msvc || compiler.id == CXX::clang_cl) {
                auto object = source;
                object.replace_extension(".obj");
                std::vector<std::string> arguments{
                    "/nologo",
                    "/std:c++20",
                    "/O2"
                };
                if (vector_math) {
                    arguments.emplace_back("/DORMI_USE_VECMATH=1");
                }
                if (direct_math) {
                    arguments.emplace_back("/DORMI_USE_DIRECT_VECMATH=1");
                }
                const auto flags = target_flags(compiler.id, target);
                arguments.insert(arguments.end(), flags.begin(), flags.end());
                const std::vector<std::string> remainder{
                    "/LD", "/EHsc",
                    "/Fo:" + object.string(),
                    source.string(),
                    "/Fe:" + output.string(),
                    "/link", "/NOIMPLIB"
                };
                arguments.insert(
                    arguments.end(),
                    remainder.begin(),
                    remainder.end()
                );
                return arguments;
            }

            std::vector<std::string> arguments{
                "-std=c++20",
                "-O3",
                "-DNDEBUG"
            };
            if (vector_math) {
                arguments.emplace_back("-DORMI_USE_VECMATH=1");
            }
            if (direct_math) {
                arguments.emplace_back("-DORMI_USE_DIRECT_VECMATH=1");
            }
            const auto flags = target_flags(compiler.id, target);
            arguments.insert(arguments.end(), flags.begin(), flags.end());
            const auto shared = platform::shared_flags();
            arguments.insert(arguments.end(), shared.begin(), shared.end());
            arguments.push_back("-o");
            arguments.push_back(output.string());
            arguments.push_back(source.string());
            return arguments;
        }
    }

    std::chrono::milliseconds compiler_timeout() {
        const char* configured = std::getenv("ORMI_CPU_JIT_TIMEOUT_MS");
        std::uintmax_t value = default_timeout_ms;

        if (configured && *configured) {
            const std::string_view text(configured);
            const auto parsed = std::from_chars(
                text.data(),
                text.data() + text.size(),
                value
            );
            if (
                parsed.ec != std::errc{} ||
                parsed.ptr != text.data() + text.size()
            ) {
                ORMI_THROW(
                    jitError,
                    "ORMI_CPU_JIT_TIMEOUT_MS must be an unsigned decimal "
                    "millisecond count."
                );
            }
        }

        if (value == 0) {
            return (std::chrono::milliseconds::max)();
        }
        using Rep = std::chrono::milliseconds::rep;
        if (value > static_cast<std::uintmax_t>(std::numeric_limits<Rep>::max())) {
            ORMI_THROW(jitError, "ORMI_CPU_JIT_TIMEOUT_MS is too large.");
        }
        return std::chrono::milliseconds(static_cast<Rep>(value));
    }

    platform::Process run(
        const Toolchain& compiler,
        std::span<const std::string> arguments
    ) {
        const auto timeout = compiler_timeout();
        auto result = platform::run(
            compiler.executable,
            arguments,
            compiler.environment,
            timeout
        );
        if (result.state == platform::Process::State::timed_out) {
            ORMI_THROW(
                jitError,
                "CPU JIT compiler timed out after " +
                    std::to_string(timeout.count()) + " ms.\nCompiler: " +
                    compiler.executable.string() + "\n" + result.output
            );
        }
        if (result.state == platform::Process::State::crashed) {
            ORMI_THROW(
                jitError,
                "CPU JIT compiler terminated abnormally with status " +
                    std::to_string(result.code) + ".\nCompiler: " +
                    compiler.executable.string() + "\n" + result.output
            );
        }
        return result;
    }

    void build(
        const Toolchain& compiler,
        Target target,
        Math functions,
        const std::filesystem::path& source,
        const std::filesystem::path& output
    ) {
        const auto result = run(
            compiler,
            command(compiler, target, functions, source, output)
        );
        auto object = source;
        object.replace_extension(".obj");
        std::error_code ignored;
        std::filesystem::remove(object, ignored);

        if (result.code == 0 && std::filesystem::is_regular_file(output)) {
            return;
        }

        ORMI_THROW(
            jitError,
            "CPU JIT compiler failed with exit code " +
                std::to_string(result.code) +
                ".\nCompiler: " + compiler.executable.string() +
                "\nTarget: " + to_string(target) +
                "\n" + result.output
        );
    }
}
