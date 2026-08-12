#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
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

        std::string lower(std::string_view value) {
            std::string result(value);
            std::transform(
                result.begin(),
                result.end(),
                result.begin(),
                [](unsigned char character) {
                    return static_cast<char>(std::tolower(character));
                }
            );
            return result;
        }

        std::optional<CXX> parse(std::string_view value) {
            const auto id = lower(value);
            if (id == "gcc") {
                return CXX::gcc;
            }
            if (id == "clang") {
                return CXX::clang;
            }
            if (id == "clang-cl" || id == "clang_cl") {
                return CXX::clang_cl;
            }
            if (id == "msvc" || id == "cl") {
                return CXX::msvc;
            }
            return std::nullopt;
        }

        std::optional<CXX> from_name(const std::filesystem::path& executable) {
            auto name = lower(executable.filename().string());
            if (name.ends_with(".exe")) {
                name.resize(name.size() - 4);
            }

            if (name == "cl") {
                return CXX::msvc;
            }
            if (name == "clang-cl") {
                return CXX::clang_cl;
            }
            if (name == "g++") {
                return CXX::gcc;
            }
            if (name == "clang++") {
                return CXX::clang;
            }
            return std::nullopt;
        }

        std::optional<CXX> from_output(std::string_view output) {
            const auto text = lower(output);
            if (text.find("clang") != std::string::npos) {
                return CXX::clang;
            }
            if (text.find("microsoft") != std::string::npos) {
                return CXX::msvc;
            }
            if (
                text.find("gcc") != std::string::npos ||
                text.find("g++") != std::string::npos ||
                text.find("free software foundation") != std::string::npos
            ) {
                return CXX::gcc;
            }
            return std::nullopt;
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

        std::optional<Toolchain> inspect_wrapper(
            const std::filesystem::path& executable,
            std::vector<std::string> environment = {}
        ) {
            const auto result = platform::run(
                executable,
                std::vector<std::string>{"--version"},
                environment,
                compiler_timeout()
            );
            if (result.state != platform::Process::State::exited) {
                return std::nullopt;
            }
            const auto id = from_output(result.output);
            return id
                ? std::optional<Toolchain>(Toolchain{
                    executable,
                    *id,
                    std::move(environment)
                })
                : std::nullopt;
        }

        std::string nonempty_environment(const char* name) {
            const char* value = std::getenv(name);
            return value && *value ? value : "";
        }

        std::vector<std::string> msvc_environment(
            const std::filesystem::path& executable
        ) {
            if (
                !nonempty_environment("INCLUDE").empty() &&
                !nonempty_environment("LIB").empty()
            ) {
                return {};
            }

            auto configured = platform::configured_msvc();
            if (!configured) {
                return {};
            }

            std::error_code error;
            if (!std::filesystem::equivalent(
                executable,
                configured->executable,
                error
            ) || error) {
                return {};
            }
            return std::move(configured->environment);
        }

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

    std::string_view to_string(CXX id) {
        switch (id) {
            case CXX::gcc:
                return "gcc";
            case CXX::clang:
                return "clang";
            case CXX::clang_cl:
                return "clang-cl";
            case CXX::msvc:
                return "msvc";
        }

        ORMI_THROW(jitError, "Unknown CPU JIT compiler id.");
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

    Toolchain resolve() {
        const auto explicit_compiler = nonempty_environment("ORMI_CPU_JIT_CXX");
        const auto explicit_id = nonempty_environment("ORMI_CPU_JIT_CXX_ID");
        std::optional<CXX> requested;

        if (!explicit_id.empty()) {
            requested = parse(explicit_id);
            if (!requested) {
                ORMI_THROW(jitError, "ORMI_CPU_JIT_CXX_ID must be gcc, clang, clang-cl, or msvc.");
            }
        }

        if (!explicit_compiler.empty()) {
            const auto executable = platform::find(explicit_compiler);
            if (!executable) {
                ORMI_THROW(
                    jitError,
                    "ORMI_CPU_JIT_CXX does not resolve to an executable: " +
                        explicit_compiler
                );
            }

            const auto named = from_name(*executable);
            const auto id = requested ? requested : named;
            auto environment = id == CXX::msvc
                ? msvc_environment(*executable)
                : std::vector<std::string>{};
            if (id) {
                return Toolchain{*executable, *id, std::move(environment)};
            }
            if (auto inspected = inspect_wrapper(
                *executable,
                std::move(environment)
            )) {
                return *inspected;
            }
            ORMI_THROW(
                jitError,
                "ORMI cannot identify the compiler syntax for " +
                    executable->string() +
                    ". Set ORMI_CPU_JIT_CXX_ID to gcc, clang, clang-cl, or msvc."
            );
        }

        for (const auto name : platform::cxx_candidates()) {
            const auto executable = platform::find(name);
            if (!executable) {
                continue;
            }

            const auto id = from_name(*executable);
            if (!id || (requested && *requested != *id)) {
                continue;
            }
            auto environment = *id == CXX::msvc
                ? msvc_environment(*executable)
                : std::vector<std::string>{};
            return Toolchain{*executable, *id, std::move(environment)};
        }

        if (!requested || *requested == CXX::msvc) {
            if (auto configured = platform::configured_msvc()) {
                return Toolchain{
                    configured->executable,
                    CXX::msvc,
                    std::move(configured->environment)
                };
            }
        }

        ORMI_THROW(
            jitError,
            "No supported CPU JIT compiler was found. Install a supported "
            "compiler or set ORMI_CPU_JIT_CXX."
        );
    }

    const Toolchain& toolchain() {
        static const Toolchain value = resolve();
        return value;
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
