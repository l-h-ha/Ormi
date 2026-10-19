#include <sstream>

#include "backends/cpu/jit/codegen/support.h"

namespace ormi::backends::cpu::jit::codegen {
    void emit_headers(
        std::ostringstream& source,
        bool float16,
        bool transcendental,
        bool intrinsics,
        bool vectorized
    ) {
        if (float16) {
            source
                << "#include <bit>\n"
                << "#include <cstdint>\n";
        }

        if (transcendental) {
            source << "#include <cmath>\n";
        }

        if (intrinsics) {
            source
                << "#if defined(ORMI_USE_F16C) || "
                << "defined(ORMI_USE_VECMATH) || "
                << "defined(ORMI_USE_AVX2)\n"
                << "#include <immintrin.h>\n"
                << "#endif\n";
        }

        source
            << "\n"
            << "#if defined(_WIN32)\n"
            << "#define ORMI_EXPORT __declspec(dllexport)\n"
            << "#else\n"
            << "#define ORMI_EXPORT "
            << "__attribute__((visibility(\"default\")))\n"
            << "#endif\n\n";

        if (vectorized) {
            source << R"ormi(#if defined(__clang__)
#define ORMI_RESTRICT __restrict
#define ORMI_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(_MSC_VER)
#define ORMI_RESTRICT __restrict
#define ORMI_VECTORIZE_LOOP __pragma(loop(ivdep))
#elif defined(__GNUC__)
#define ORMI_RESTRICT __restrict__
#define ORMI_VECTORIZE_LOOP _Pragma("GCC ivdep")
#else
#define ORMI_RESTRICT
#define ORMI_VECTORIZE_LOOP
#endif

template <typename T>
inline T* ormi_assume_aligned(T* pointer) noexcept {
#if defined(__clang__) || defined(__GNUC__)
    return static_cast<T*>(__builtin_assume_aligned(pointer, 64));
#elif defined(_MSC_VER)
    __assume((reinterpret_cast<unsigned long long>(pointer) & 63ull) == 0);
    return pointer;
#else
    return pointer;
#endif
}

)ormi";
        }
    }

    void emit_float16_support(std::ostringstream& source) {
        source << R"(
namespace {
    std::uint16_t ormi_half_bits(float value) noexcept {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        const auto sign = (bits >> 16) & 0x8000u;
        const auto exponent = (bits >> 23) & 0xffu;
        auto mantissa = bits & 0x007fffffu;

        if (exponent == 0xffu) {
            return static_cast<std::uint16_t>(
                sign | (mantissa == 0 ? 0x7c00u : 0x7e00u)
            );
        }

        const auto half_exponent = static_cast<std::int32_t>(exponent) - 112;
        if (half_exponent >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        if (half_exponent <= 0) {
            if (half_exponent < -10) {
                return static_cast<std::uint16_t>(sign);
            }

            mantissa |= 0x00800000u;
            const auto shift = static_cast<std::uint32_t>(14 - half_exponent);
            auto half_mantissa = mantissa >> shift;
            const auto remainder = mantissa & ((1u << shift) - 1u);
            const auto halfway = 1u << (shift - 1u);
            if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) {
                ++half_mantissa;
            }
            return static_cast<std::uint16_t>(sign | half_mantissa);
        }

        auto half = sign |
            (static_cast<std::uint32_t>(half_exponent) << 10) |
            (mantissa >> 13);
        const auto remainder = mantissa & 0x1fffu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
            ++half;
        }
        return static_cast<std::uint16_t>(half);
    }

    float ormi_half_value(std::uint16_t value) noexcept {
        const auto sign = (static_cast<std::uint32_t>(value) & 0x8000u) << 16;
        const auto exponent = (static_cast<std::uint32_t>(value) >> 10) & 0x1fu;
        auto mantissa = static_cast<std::uint32_t>(value) & 0x03ffu;
        std::uint32_t bits;

        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                std::int32_t normalized_exponent = -14;
                while ((mantissa & 0x0400u) == 0) {
                    mantissa <<= 1;
                    --normalized_exponent;
                }
                mantissa &= 0x03ffu;
                bits = sign |
                    (static_cast<std::uint32_t>(normalized_exponent + 127) << 23) |
                    (mantissa << 13);
            }
        } else if (exponent == 0x1fu) {
            bits = sign | 0x7f800000u | (mantissa << 13);
        } else {
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
        return std::bit_cast<float>(bits);
    }

    float ormi_half_round(float value) noexcept {
        return ormi_half_value(ormi_half_bits(value));
    }
}

)";
        source << R"(#if defined(ORMI_USE_F16C)
namespace {
    __m128i ormi_half_bits8(__m256 value) noexcept {
        auto bits = _mm256_cvtps_ph(
            value,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
        );
        const auto sign = _mm_and_si128(bits, _mm_set1_epi16(-32768));
        const auto magnitude = _mm_and_si128(bits, _mm_set1_epi16(0x7fff));
        const auto nan = _mm_cmpgt_epi16(magnitude, _mm_set1_epi16(0x7c00));
        const auto canonical = _mm_or_si128(sign, _mm_set1_epi16(0x7e00));
        return _mm_blendv_epi8(bits, canonical, nan);
    }

    __m256 ormi_half_round8(__m256 value) noexcept {
        return _mm256_cvtph_ps(ormi_half_bits8(value));
    }

    __m256 ormi_half_load8(const std::uint16_t* values) noexcept {
        return _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(values))
        );
    }
}
#endif

)";
    }

    void emit_math_support(std::ostringstream& source) {
        source << R"(
#if defined(ORMI_USE_DIRECT_VECMATH)
namespace {
    using ormi_unary8 = __m256 (*)(__m256);
#if defined(_MSC_VER)
    struct ormi_sincos8_result {
        __m256 sine;
        __m256 cosine;
    };
    using ormi_sincos8_fn = ormi_sincos8_result (__vectorcall *)(__m256);
#else
    using ormi_sincos8_fn = void (*)(__m256*, __m256*, __m256);
#endif

    struct ormi_math_table {
        ormi_unary8 unary[4];
        ormi_sincos8_fn sincos;
    };

    __m256 ormi_math8(
        const void* functions,
        unsigned int operation,
        __m256 input
    ) {
        const auto* math = static_cast<const ormi_math_table*>(functions);
        return math->unary[operation](input);
    }

    void ormi_sincos8(
        const void* functions,
        __m256 input,
        __m256& sine,
        __m256& cosine
    ) {
        const auto* math = static_cast<const ormi_math_table*>(functions);
#if defined(_MSC_VER)
        const auto result = math->sincos(input);
        sine = result.sine;
        cosine = result.cosine;
#else
        math->sincos(&sine, &cosine, input);
#endif
    }
}
#elif defined(ORMI_USE_VECMATH)
namespace {
    using ormi_unary8 = void (*)(float*, const float*);
    using ormi_sincos8_fn = void (*)(float*, float*, const float*);

    struct ormi_math_table {
        ormi_unary8 unary[4];
        ormi_sincos8_fn sincos;
    };

    __m256 ormi_math8(
        const void* functions,
        unsigned int operation,
        __m256 input
    ) {
        const auto* math = static_cast<const ormi_math_table*>(functions);
        alignas(32) float buffered_input[8];
        alignas(32) float buffered_output[8];
        _mm256_storeu_ps(buffered_input, input);
        math->unary[operation](buffered_output, buffered_input);
        return _mm256_loadu_ps(buffered_output);
    }

    void ormi_sincos8(
        const void* functions,
        __m256 input,
        __m256& sine,
        __m256& cosine
    ) {
        const auto* math = static_cast<const ormi_math_table*>(functions);
        alignas(32) float buffered_input[8];
        alignas(32) float buffered_sine[8];
        alignas(32) float buffered_cosine[8];
        _mm256_storeu_ps(buffered_input, input);
        math->sincos(buffered_sine, buffered_cosine, buffered_input);
        sine = _mm256_loadu_ps(buffered_sine);
        cosine = _mm256_loadu_ps(buffered_cosine);
    }
}
#endif

)";
    }
}
