#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/math.h"
#include "backends/cpu/jit/target.h"
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

        bool compatible_vector_abi(CXX compiler) noexcept {
#if defined(_WIN32)
#if defined(_MSC_VER)
            return compiler == CXX::msvc || compiler == CXX::clang_cl;
#elif defined(__clang__)
            return compiler == CXX::clang;
#elif defined(__GNUC__)
            return compiler == CXX::gcc;
#else
            return false;
#endif
#else
            return compiler == CXX::gcc || compiler == CXX::clang;
#endif
        }
    }

    Target select_target() {
        const char* configured = std::getenv("ORMI_CPU_JIT_TARGET");
        const auto requested = lower(configured && *configured ? configured : "auto");
        const bool avx2 = platform::supports_avx2();
        const bool f16c = avx2 && platform::supports_f16c();

        if (requested == "auto") {
            return Target{.avx2 = avx2, .f16c = f16c};
        }
        if (requested == "baseline") {
            return {};
        }
        if (requested == "avx2") {
            if (!avx2) {
                ORMI_THROW(
                    jitError,
                    "ORMI_CPU_JIT_TARGET requests AVX2, but this CPU or operating "
                    "system does not support AVX2."
                );
            }
            return Target{.avx2 = true, .f16c = f16c};
        }

        ORMI_THROW(
            jitError,
            "ORMI_CPU_JIT_TARGET must be auto, baseline, or avx2."
        );
    }

    std::string to_string(Target target) {
        if (!target.avx2 && !target.f16c) {
            return "baseline";
        }
        if (target.avx2) {
            return target.f16c ? "avx2+f16c" : "avx2";
        }

        ORMI_THROW(jitError, "F16C cannot be selected without AVX2.");
    }

    std::vector<std::string> target_flags(CXX compiler, Target target) {
        if (!target.avx2 && !target.f16c) {
            return {};
        }
        if (!target.avx2) {
            ORMI_THROW(jitError, "F16C cannot be selected without AVX2.");
        }

        if (compiler == CXX::gcc || compiler == CXX::clang) {
            std::vector<std::string> result{
                "-mavx2",
                "-DORMI_USE_AVX2=1"
            };
            if (target.f16c) {
                result.emplace_back("-mf16c");
                result.emplace_back("-DORMI_USE_F16C=1");
            }
            return result;
        }
        std::vector<std::string> result{
            "/arch:AVX2",
            "/DORMI_USE_AVX2=1"
        };
        if (target.f16c) {
            result.emplace_back("/DORMI_USE_F16C=1");
        }
        return result;
    }

    const Target& selected_target() {
        static const Target value = select_target();
        return value;
    }

    Math selected_math(CXX compiler) noexcept {
        if (!selected_target().avx2) {
            return {};
        }
        return math(
            compatible_vector_abi(compiler)
                ? MathABI::direct
                : MathABI::buffered
        );
    }

    std::string context(Math functions) {
        auto result = to_string(selected_target());
        if (!functions.functions) {
            return result + "+scalar-math";
        }
        return result + (
            functions.abi == MathABI::direct
                ? "+sleef-direct-3.9.0"
                : "+sleef-buffered-3.9.0"
        );
    }

    std::string_view target_context() {
        static const std::string value = to_string(selected_target());
        return value;
    }
}
