#include <sstream>

#include <gtest/gtest.h>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>

namespace {
    TEST(DTypeTest, mapsDTypesToByteSizes) {
        EXPECT_EQ(ormi::core::dtype_size(ormi::core::DType::float64), 8u);
        EXPECT_EQ(ormi::core::dtype_size(ormi::core::DType::float32), 4u);
        EXPECT_EQ(ormi::core::dtype_size(ormi::core::DType::float16), 2u);
        EXPECT_EQ(ormi::core::dtype_size(ormi::core::DType::int32), 4u);
        EXPECT_EQ(ormi::core::dtype_size(ormi::core::DType::boolean), 1u);
    }

    TEST(DTypeTest, mapsDTypesToStableNames) {
        EXPECT_EQ(ormi::core::to_string(ormi::core::DType::float64), "float64");
        EXPECT_EQ(ormi::core::to_string(ormi::core::DType::float32), "float32");
        EXPECT_EQ(ormi::core::to_string(ormi::core::DType::float16), "float16");
        EXPECT_EQ(ormi::core::to_string(ormi::core::DType::int32), "int32");
        EXPECT_EQ(ormi::core::to_string(ormi::core::DType::boolean), "bool");
    }

    TEST(DTypeTest, classifiesDTypes) {
        EXPECT_TRUE(ormi::core::is_floating(ormi::core::DType::float64));
        EXPECT_TRUE(ormi::core::is_floating(ormi::core::DType::float32));
        EXPECT_TRUE(ormi::core::is_floating(ormi::core::DType::float16));
        EXPECT_FALSE(ormi::core::is_floating(ormi::core::DType::int32));
        EXPECT_FALSE(ormi::core::is_floating(ormi::core::DType::boolean));

        EXPECT_TRUE(ormi::core::is_integer(ormi::core::DType::int32));
        EXPECT_FALSE(ormi::core::is_integer(ormi::core::DType::float32));
        EXPECT_FALSE(ormi::core::is_integer(ormi::core::DType::boolean));

        EXPECT_TRUE(ormi::core::is_boolean(ormi::core::DType::boolean));
        EXPECT_FALSE(ormi::core::is_boolean(ormi::core::DType::float32));
        EXPECT_FALSE(ormi::core::is_boolean(ormi::core::DType::int32));

        EXPECT_TRUE(ormi::core::is_numeric(ormi::core::DType::float64));
        EXPECT_TRUE(ormi::core::is_numeric(ormi::core::DType::float32));
        EXPECT_TRUE(ormi::core::is_numeric(ormi::core::DType::float16));
        EXPECT_TRUE(ormi::core::is_numeric(ormi::core::DType::int32));
        EXPECT_FALSE(ormi::core::is_numeric(ormi::core::DType::boolean));
    }

    TEST(DTypeTest, promotesFloatingDTypesSymmetrically) {
        using ormi::core::DType;

        EXPECT_EQ(ormi::core::promote(DType::float16, DType::float16), DType::float16);
        EXPECT_EQ(ormi::core::promote(DType::float16, DType::float32), DType::float32);
        EXPECT_EQ(ormi::core::promote(DType::float32, DType::float16), DType::float32);
        EXPECT_EQ(ormi::core::promote(DType::float16, DType::float64), DType::float64);
        EXPECT_EQ(ormi::core::promote(DType::float64, DType::float16), DType::float64);
        EXPECT_EQ(ormi::core::promote(DType::float32, DType::float64), DType::float64);
        EXPECT_EQ(ormi::core::promote(DType::float64, DType::float32), DType::float64);
        EXPECT_EQ(ormi::core::promote(DType::int32, DType::int32), DType::int32);
    }

    TEST(DTypeTest, rejectsUnsafeOrBooleanPromotion) {
        using ormi::core::DType;

        EXPECT_THROW(
            (void)ormi::core::promote(DType::int32, DType::float32),
            ormi::DTypeError
        );
        EXPECT_THROW(
            (void)ormi::core::promote(DType::float16, DType::int32),
            ormi::DTypeError
        );
        EXPECT_THROW(
            (void)ormi::core::promote(DType::boolean, DType::boolean),
            ormi::DTypeError
        );
    }

    TEST(DTypeTest, streamsDTypeName) {
        std::ostringstream out;
        out << ormi::core::DType::float32;
        EXPECT_EQ(out.str(), "float32");
    }

    TEST(DTypeTest, rejectsUnknownDType) {
        const auto dtype = static_cast<ormi::core::DType>(999);
        EXPECT_THROW((void)ormi::core::dtype_size(dtype), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::to_string(dtype), ormi::DTypeError);
        EXPECT_THROW(
            (void)ormi::core::promote(dtype, ormi::core::DType::float32),
            ormi::DTypeError
        );
        EXPECT_FALSE(ormi::core::is_floating(dtype));
        EXPECT_FALSE(ormi::core::is_integer(dtype));
        EXPECT_FALSE(ormi::core::is_boolean(dtype));
        EXPECT_FALSE(ormi::core::is_numeric(dtype));
    }
}
