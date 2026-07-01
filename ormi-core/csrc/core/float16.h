#pragma once

#include <cstdint>
#include <cstring>

namespace ormi::fp16 {
    template <typename To, typename From>
    inline To bit_cast(const From& value) {
        static_assert(sizeof(To) == sizeof(From), "bit_cast requires equal sizes.");

        To out;
        std::memcpy(&out, &value, sizeof(To));
        return out;
    }

    inline std::uint16_t float_to_bits(float value) {
        std::uint32_t bits = bit_cast<std::uint32_t>(value);

        std::uint32_t sign = (bits >> 16) & 0x8000u;
        std::uint32_t exp = (bits >> 23) & 0xffu;
        std::uint32_t mant = bits & 0x007fffffu;

        if (exp == 0xffu) {
            if (mant == 0) {
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            }

            return static_cast<std::uint16_t>(sign | 0x7e00u);
        }

        std::int32_t new_exp = static_cast<std::int32_t>(exp) - 127 + 15;

        if (new_exp >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }

        if (new_exp <= 0) {
            if (new_exp < -10) {
                return static_cast<std::uint16_t>(sign);
            }

            mant |= 0x00800000u;

            std::int32_t shift = 14 - new_exp;
            std::uint32_t half_mant = mant >> shift;

            std::uint32_t remainder_mask = (1u << shift) - 1u;
            std::uint32_t remainder = mant & remainder_mask;
            std::uint32_t halfway = 1u << (shift - 1);

            if (
                remainder > halfway ||
                (remainder == halfway && (half_mant & 1u))
            ) {
                half_mant++;
            }

            return static_cast<std::uint16_t>(sign | half_mant);
        }

        std::uint32_t half =
            sign |
            (static_cast<std::uint32_t>(new_exp) << 10) |
            (mant >> 13);

        std::uint32_t remainder = mant & 0x1fffu;
        std::uint32_t halfway = 0x1000u;

        if (
            remainder > halfway ||
            (remainder == halfway && (half & 1u))
        ) {
            half++;
        }

        return static_cast<std::uint16_t>(half);
    }

    inline float bits_to_float(std::uint16_t value) {
        std::uint32_t sign =
            (static_cast<std::uint32_t>(value) & 0x8000u) << 16;
        std::uint32_t exp =
            (static_cast<std::uint32_t>(value) >> 10) & 0x1fu;
        std::uint32_t mant =
            static_cast<std::uint32_t>(value) & 0x03ffu;

        std::uint32_t out = 0;

        if (exp == 0) {
            if (mant == 0) {
                out = sign;
            } else {
                exp = 1;

                while ((mant & 0x0400u) == 0) {
                    mant <<= 1;
                    exp--;
                }

                mant &= 0x03ffu;

                out =
                    sign |
                    ((exp + (127 - 15)) << 23) |
                    (mant << 13);
            }
        } else if (exp == 0x1fu) {
            out = sign | 0x7f800000u | (mant << 13);
        } else {
            out =
                sign |
                ((exp + (127 - 15)) << 23) |
                (mant << 13);
        }

        return bit_cast<float>(out);
    }
}
