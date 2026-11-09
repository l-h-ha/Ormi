#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/compatibility.h"
#include "backends/cpu/jit/compiler.h"
#include "platform/platform.h"

namespace ormi::backends::cpu::jit {
    namespace {
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

        std::optional<Toolchain> identify(
            const std::filesystem::path& executable,
            std::optional<CXX> requested = std::nullopt
        ) {
            const auto named = from_name(executable);
            const auto id = requested ? requested : named;
            if (!id) {
                return inspect_wrapper(executable);
            }

            std::vector<std::string> environment;
            if (*id == CXX::msvc) {
                environment = msvc_environment(executable);
            }
            return Toolchain{executable, *id, std::move(environment)};
        }

        void append_rejection(
            std::string& diagnostics,
            const Toolchain& compiler,
            std::string_view reason
        ) {
            diagnostics += "\n- " + compiler.executable.string() + " (" +
                std::string(to_string(compiler.id)) + "): " +
                std::string(reason);
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
                    CompilerUnavailable,
                    "ORMI_CPU_JIT_CXX does not resolve to an executable: " +
                        explicit_compiler
                );
            }

            if (auto compiler = identify(*executable, requested)) {
                return std::move(*compiler);
            }
            ORMI_THROW(
                jitError,
                "ORMI cannot identify the compiler syntax for " +
                    executable->string() +
                    ". Set ORMI_CPU_JIT_CXX_ID to gcc, clang, clang-cl, or msvc."
            );
        }

        std::string diagnostics;
        for (const auto name : platform::cxx_candidates()) {
            const auto executable = platform::find(name);
            if (!executable) {
                continue;
            }

            auto compiler = identify(*executable);
            if (!compiler) {
                continue;
            }
            if (requested && *requested != compiler->id) {
                continue;
            }
            if (const auto reason = compatibility::check(
                *compiler,
                compiler_timeout()
            )) {
                append_rejection(diagnostics, *compiler, *reason);
                continue;
            }
            return std::move(*compiler);
        }

        if (!requested || *requested == CXX::msvc) {
            if (auto configured = platform::configured_msvc()) {
                Toolchain compiler{
                    configured->executable,
                    CXX::msvc,
                    std::move(configured->environment)
                };
                if (const auto reason = compatibility::check(
                    compiler,
                    compiler_timeout()
                )) {
                    append_rejection(diagnostics, compiler, *reason);
                } else {
                    return compiler;
                }
            }
        }

        ORMI_THROW(
            CompilerUnavailable,
            "No compatible CPU JIT compiler was found. Install a supported "
            "compiler or set ORMI_CPU_JIT_CXX." +
                (diagnostics.empty()
                    ? std::string{}
                    : "\nRejected candidates:" + diagnostics)
        );
    }

    const Toolchain& toolchain() {
        static const Toolchain value = resolve();
        return value;
    }

    bool compiler_available() {
        static const bool available = [] {
            try {
                (void)toolchain();
                return true;
            } catch (const CompilerUnavailable&) {
                return false;
            }
        }();
        return available;
    }
}
