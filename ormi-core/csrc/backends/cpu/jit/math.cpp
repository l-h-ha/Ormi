#include <cstddef>
#include <optional>
#include <variant>

#if defined(ORMI_ENABLE_SLEEF)
#include <immintrin.h>
#include <sleef.h>
#endif

#include <ormi/core/ops/kind.h>

#include "backends/cpu/jit/math.h"

#if defined(ORMI_ENABLE_SLEEF)
#include "platform/platform.h"
#endif

namespace ormi::backends::cpu::jit {
    std::optional<core::runtime::fusion::Index> trig_pair(
        const core::runtime::fusion::Expr& expression,
        std::size_t index
    ) noexcept {
        using core::runtime::fusion::Index;
        using core::runtime::fusion::Unary;

        const auto* current = std::get_if<Unary>(
            &expression.nodes[index].instruction
        );
        if (
            !current ||
            (
                current->op != core::ops::UnaryOp::sin &&
                current->op != core::ops::UnaryOp::cos
            )
        ) {
            return std::nullopt;
        }

        std::size_t occurrence = 0;
        for (std::size_t i = 0; i < index; ++i) {
            const auto* prior = std::get_if<Unary>(
                &expression.nodes[i].instruction
            );
            if (
                prior &&
                prior->op == current->op &&
                prior->input == current->input
            ) {
                ++occurrence;
            }
        }

        const auto opposite = current->op == core::ops::UnaryOp::sin
            ? core::ops::UnaryOp::cos
            : core::ops::UnaryOp::sin;
        for (std::size_t i = 0; i < expression.nodes.size(); ++i) {
            const auto* candidate = std::get_if<Unary>(
                &expression.nodes[i].instruction
            );
            if (
                !candidate ||
                candidate->op != opposite ||
                candidate->input != current->input ||
                expression.nodes[i].dtype != expression.nodes[index].dtype
            ) {
                continue;
            }
            if (occurrence == 0) {
                return static_cast<Index>(i);
            }
            --occurrence;
        }
        return std::nullopt;
    }

#if defined(ORMI_ENABLE_SLEEF)
    namespace {
        using BufferedSincos8 = void (*)(float*, float*, const float*);
        struct BufferedMath {
            std::array<Unary8, 4> unary;
            BufferedSincos8 sincos;
        };

        using DirectUnary8 = __m256 (*)(__m256);
#if defined(_MSC_VER)
        struct DirectSincosResult {
            __m256 sine;
            __m256 cosine;
        };
        using DirectSincos8 = DirectSincosResult (__vectorcall *)(__m256);
#else
        using DirectSincos8 = void (*)(__m256*, __m256*, __m256);
#endif
        struct DirectMath {
            std::array<DirectUnary8, 4> unary;
            DirectSincos8 sincos;
        };

        void exp8(float* output, const float* input) {
            _mm256_storeu_ps(
                output,
                Sleef_expf8_u10avx2(_mm256_loadu_ps(input))
            );
        }

        void log8(float* output, const float* input) {
            _mm256_storeu_ps(
                output,
                Sleef_logf8_u10avx2(_mm256_loadu_ps(input))
            );
        }

        void sin8(float* output, const float* input) {
            _mm256_storeu_ps(
                output,
                Sleef_sinf8_u35avx2(_mm256_loadu_ps(input))
            );
        }

        void cos8(float* output, const float* input) {
            _mm256_storeu_ps(
                output,
                Sleef_cosf8_u35avx2(_mm256_loadu_ps(input))
            );
        }

        void sincos8(
            float* sine,
            float* cosine,
            const float* input
        ) {
            const auto result = Sleef_sincosf8_u35avx2(
                _mm256_loadu_ps(input)
            );
            _mm256_storeu_ps(sine, result.x);
            _mm256_storeu_ps(cosine, result.y);
        }

#if defined(_MSC_VER)
        DirectSincosResult __vectorcall direct_sincos8(__m256 input) {
            const auto result = Sleef_sincosf8_u35avx2(input);
            return {result.x, result.y};
        }
#else
        void direct_sincos8(
            __m256* sine,
            __m256* cosine,
            __m256 input
        ) {
            const auto result = Sleef_sincosf8_u35avx2(input);
            *sine = result.x;
            *cosine = result.y;
        }
#endif

        const BufferedMath buffered{
            {exp8, log8, sin8, cos8},
            sincos8
        };
        const DirectMath direct{
            {
                Sleef_expf8_u10avx2,
                Sleef_logf8_u10avx2,
                Sleef_sinf8_u35avx2,
                Sleef_cosf8_u35avx2
            },
            direct_sincos8
        };
    }
#endif

    Math math([[maybe_unused]] MathABI abi) noexcept {
#if defined(ORMI_ENABLE_SLEEF)
        if (platform::supports_avx2()) {
            return abi == MathABI::direct
                ? Math{&direct, abi}
                : Math{&buffered, abi};
        }
#endif
        return {};
    }
}
