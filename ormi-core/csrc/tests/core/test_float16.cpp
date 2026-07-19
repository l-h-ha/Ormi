#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <ormi/core/float16.h>

namespace {
    using ormi::core::Float16;

    TEST(Float16Test, storesExactlyTwoBytes) {
        EXPECT_EQ(sizeof(Float16), sizeof(std::uint16_t));
    }

    TEST(Float16Test, encodesKnownValues) {
        EXPECT_EQ(Float16(0.0f).bits(), 0x0000u);
        EXPECT_EQ(Float16(-0.0f).bits(), 0x8000u);
        EXPECT_EQ(Float16(1.0f).bits(), 0x3c00u);
        EXPECT_EQ(Float16(-2.0f).bits(), 0xc000u);
        EXPECT_EQ(Float16(65504.0f).bits(), 0x7bffu);
        EXPECT_EQ(Float16(std::numeric_limits<float>::infinity()).bits(), 0x7c00u);
        EXPECT_EQ(Float16(-std::numeric_limits<float>::infinity()).bits(), 0xfc00u);
    }

    TEST(Float16Test, encodesNormalAndSubnormalLimits) {
        EXPECT_EQ(Float16(std::ldexp(1.0f, -14)).bits(), 0x0400u);
        EXPECT_EQ(Float16(std::ldexp(1.0f, -24)).bits(), 0x0001u);
    }

    TEST(Float16Test, roundsTiesToEven) {
        EXPECT_EQ(Float16(1.0f + std::ldexp(1.0f, -11)).bits(), 0x3c00u);
        EXPECT_EQ(Float16(1.0f + std::ldexp(3.0f, -11)).bits(), 0x3c02u);
    }

    TEST(Float16Test, decodesKnownValues) {
        EXPECT_FLOAT_EQ(static_cast<float>(Float16::from_bits(0x0000u)), 0.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(Float16::from_bits(0x3c00u)), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(Float16::from_bits(0xc000u)), -2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(Float16::from_bits(0x0400u)), std::ldexp(1.0f, -14));
        EXPECT_FLOAT_EQ(static_cast<float>(Float16::from_bits(0x0001u)), std::ldexp(1.0f, -24));
    }

    TEST(Float16Test, preservesSpecialValues) {
        const auto quiet_nan = Float16(std::numeric_limits<float>::quiet_NaN());
        EXPECT_EQ(quiet_nan.bits() & 0x7c00u, 0x7c00u);
        EXPECT_NE(quiet_nan.bits() & 0x03ffu, 0u);
        EXPECT_TRUE(std::isnan(static_cast<float>(quiet_nan)));
        EXPECT_TRUE(std::isinf(static_cast<float>(Float16::from_bits(0x7c00u))));
        EXPECT_TRUE(std::signbit(static_cast<float>(Float16::from_bits(0x8000u))));
    }

    TEST(Float16Test, arithmeticRoundsStoredResults) {
        const auto lhs = Float16(1.5f);
        const auto rhs = Float16(0.5f);

        EXPECT_FLOAT_EQ(static_cast<float>(lhs + rhs), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(lhs - rhs), 1.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(lhs * rhs), 0.75f);
        EXPECT_FLOAT_EQ(static_cast<float>(lhs / rhs), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(-lhs), -1.5f);
    }
}
