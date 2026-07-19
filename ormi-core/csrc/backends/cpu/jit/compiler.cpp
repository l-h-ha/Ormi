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

        std::vector<std::string> probe_arguments(CXX id) {
            return id == CXX::msvc
                ? std::vector<std::string>{"/Bv"}
                : std::vector<std::string>{"--version"};
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

        std::optional<Toolchain> probe(
            const std::filesystem::path& executable,
            std::optional<CXX> requested,
            bool trust_requested,
            std::vector<std::string> environment = {}
        ) {
            const auto named = from_name(executable);
            if (requested && !trust_requested && named && *requested != *named) {
                return std::nullopt;
            }

            auto id = requested ? requested : named;

            if (id) {
                const auto result = platform::run(
                    executable,
                    probe_arguments(*id),
                    environment,
                    compiler_timeout()
                );
                if (result.state != platform::Process::State::exited) {
                    return std::nullopt;
                }
                if (result.code == 0 || !result.output.empty()) {
                    if (!trust_requested) {
                        const auto detected = named ? named : from_output(result.output);
                        if (!detected || *detected != *id) {
                            return std::nullopt;
                        }
                    }
                    return Toolchain{executable, *id, std::move(environment)};
                }
                return std::nullopt;
            }

            const auto result = platform::run(
                executable,
                std::vector<std::string>{"--version"},
                environment,
                compiler_timeout()
            );
            if (result.state != platform::Process::State::exited) {
                return std::nullopt;
            }
            id = from_output(result.output);
            return id
                ? std::optional<Toolchain>(Toolchain{executable, *id, std::move(environment)})
                : std::nullopt;
        }

        std::string nonempty_environment(const char* name) {
            const char* value = std::getenv(name);
            return value && *value ? value : "";
        }

        std::vector<std::string> msvc_environment(
            const std::filesystem::path& executable
        ) {
#if defined(_WIN32)
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
#else
            (void)executable;
            return {};
#endif
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

            auto environment =
                (requested == CXX::msvc || from_name(*executable) == CXX::msvc)
                ? msvc_environment(*executable)
                : std::vector<std::string>{};
            const auto toolchain = probe(
                *executable,
                requested,
                requested.has_value(),
                std::move(environment)
            );
            if (!toolchain) {
                ORMI_THROW(
                    jitError,
                    "ORMI_CPU_JIT_CXX is not a usable C++ compiler: " +
                        executable->string()
                );
            }
            return *toolchain;
        }

        for (const auto name : platform::cxx_candidates()) {
            const auto executable = platform::find(name);
            if (!executable) {
                continue;
            }

            auto environment = from_name(*executable) == CXX::msvc
                ? msvc_environment(*executable)
                : std::vector<std::string>{};
            const auto toolchain = probe(
                *executable,
                requested,
                false,
                std::move(environment)
            );
            if (toolchain) {
                return *toolchain;
            }
        }

        if (!requested || *requested == CXX::msvc) {
            if (auto configured = platform::configured_msvc()) {
                if (auto toolchain = probe(
                    configured->executable,
                    CXX::msvc,
                    false,
                    std::move(configured->environment)
                )) {
                    return *toolchain;
                }
            }
        }

        ORMI_THROW(
            jitError,
            "No supported CPU JIT compiler was found. Install a supported "
            "compiler or set ORMI_CPU_JIT_CXX."
        );
    }
}
