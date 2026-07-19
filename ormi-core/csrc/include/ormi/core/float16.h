#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

namespace ormi::core {
    namespace detail {
        inline std::uint16_t float16_bits(float value) noexcept {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            const auto sign = (bits >> 16) & 0x8000u;
            const auto exponent = (bits >> 23) & 0xffu;
            auto mantissa = bits & 0x007fffffu;

            if (exponent == 0xffu) {
                if (mantissa == 0) {
                    return static_cast<std::uint16_t>(sign | 0x7c00u);
                }

                return static_cast<std::uint16_t>(sign | 0x7e00u);
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

        inline float float16_value(std::uint16_t value) noexcept {
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
    }

    class Float16 {
    public:
        constexpr Float16() noexcept = default;
        explicit Float16(float value) noexcept : bits_(detail::float16_bits(value)) {}

        static constexpr Float16 from_bits(std::uint16_t bits) noexcept {
            Float16 value;
            value.bits_ = bits;
            return value;
        }

        constexpr std::uint16_t bits() const noexcept {
            return bits_;
        }

        explicit operator float() const noexcept {
            return detail::float16_value(bits_);
        }

        friend Float16 operator+(Float16 lhs, Float16 rhs) noexcept {
            return Float16(static_cast<float>(lhs) + static_cast<float>(rhs));
        }

        friend Float16 operator-(Float16 lhs, Float16 rhs) noexcept {
            return Float16(static_cast<float>(lhs) - static_cast<float>(rhs));
        }

        friend Float16 operator*(Float16 lhs, Float16 rhs) noexcept {
            return Float16(static_cast<float>(lhs) * static_cast<float>(rhs));
        }

        friend Float16 operator/(Float16 lhs, Float16 rhs) noexcept {
            return Float16(static_cast<float>(lhs) / static_cast<float>(rhs));
        }

        friend Float16 operator-(Float16 value) noexcept {
            return Float16(-static_cast<float>(value));
        }

    private:
        std::uint16_t bits_ = 0;
    };

    static_assert(sizeof(Float16) == sizeof(std::uint16_t));
    static_assert(std::is_trivially_copyable_v<Float16>);
}
