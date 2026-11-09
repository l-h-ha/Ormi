#include <array>
#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    TEST(CastTest, constructsLazyDTypeChangingTensor) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3},
            ormi::core::DType::float32
        );
        const auto result = ormi::core::ops::cast(input, ormi::core::DType::float16);

        EXPECT_FALSE(result->realized());
        EXPECT_EQ(result->dtype(), ormi::core::DType::float16);
        EXPECT_EQ(result->shape(), input->shape());
        ASSERT_NE(result->op(), nullptr);
        EXPECT_EQ(result->op()->name, "cast");
        ASSERT_EQ(result->parents().size(), 1u);
        EXPECT_EQ(result->parents()[0], input);
    }

    TEST(CastTest, returnsInputWhenDTypeAlreadyMatches) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float32
        );

        EXPECT_EQ(ormi::core::ops::cast(input, ormi::core::DType::float32), input);
    }

    TEST(CPUCastTest, convertsFloat32ToFloat16AndBack) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3},
            ormi::core::DType::float32
        );
        auto* input_data = static_cast<float*>(input->data());
        input_data[0] = -1.25f;
        input_data[1] = 0.0f;
        input_data[2] = 2.5f;

        const auto half = ormi::core::ops::cast(input, ormi::core::DType::float16);
        const auto restored = ormi::core::ops::cast(half, ormi::core::DType::float32);
        const auto* data = static_cast<const float*>(restored->data());

        EXPECT_FLOAT_EQ(data[0], -1.25f);
        EXPECT_FLOAT_EQ(data[1], 0.0f);
        EXPECT_FLOAT_EQ(data[2], 2.5f);
    }

    TEST(CPUCastTest, convertsIntegerAndBooleanTensors) {
        const auto integers = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3},
            ormi::core::DType::int32
        );
        auto* values = static_cast<std::int32_t*>(integers->data());
        values[0] = -2;
        values[1] = 0;
        values[2] = 3;

        const auto booleans = ormi::core::ops::cast(integers, ormi::core::DType::boolean);
        const auto restored = ormi::core::ops::cast(booleans, ormi::core::DType::float32);
        const auto* data = static_cast<const float*>(restored->data());

        EXPECT_FLOAT_EQ(data[0], 1.0f);
        EXPECT_FLOAT_EQ(data[1], 0.0f);
        EXPECT_FLOAT_EQ(data[2], 1.0f);
    }

    TEST(CPUCastTest, readsTransposedInputByItsStrides) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(i + 1);
        }

        const auto transposed = ormi::core::ops::transpose(input, {1, 0});
        const auto result = ormi::core::ops::cast(transposed, ormi::core::DType::float64);
        const auto* data = static_cast<const double*>(result->data());

        EXPECT_DOUBLE_EQ(data[0], 1.0);
        EXPECT_DOUBLE_EQ(data[1], 4.0);
        EXPECT_DOUBLE_EQ(data[2], 2.0);
        EXPECT_DOUBLE_EQ(data[3], 5.0);
        EXPECT_DOUBLE_EQ(data[4], 3.0);
        EXPECT_DOUBLE_EQ(data[5], 6.0);
    }

    TEST(CPUCastTest, carriesFloat16ComputationBackToFloat32Parameter) {
        const auto parameter = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float32
        );
        *static_cast<float*>(parameter->data()) = 3.0f;
        const auto half = ormi::core::ops::cast(parameter, ormi::core::DType::float16);
        const auto output = ormi::core::ops::mul(half, half);
        const auto gradient = ormi::core::grad(output, parameter);
        const auto second = ormi::core::grad(gradient, parameter);

        EXPECT_EQ(gradient->dtype(), ormi::core::DType::float32);
        EXPECT_EQ(second->dtype(), ormi::core::DType::float32);
        EXPECT_FALSE(gradient->realized());
        EXPECT_FALSE(second->realized());
        EXPECT_FLOAT_EQ(*static_cast<const float*>(gradient->data()), 6.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(second->data()), 2.0f);
    }

    TEST(CPUCastTest, preservesFloat64PrecisionWhenCastingToInt32) {
        const std::array values{
            static_cast<double>(std::numeric_limits<std::int32_t>::min()),
            static_cast<double>(std::numeric_limits<std::int32_t>::max()),
            static_cast<double>(std::numeric_limits<std::int32_t>::min()) + 0.25,
            static_cast<double>(std::numeric_limits<std::int32_t>::max()) - 0.25,
            -2.75,
            3.9
        };
        const std::array expected{
            std::numeric_limits<std::int32_t>::min(),
            std::numeric_limits<std::int32_t>::max(),
            std::numeric_limits<std::int32_t>::min() + 1,
            std::numeric_limits<std::int32_t>::max() - 1,
            std::int32_t{-2},
            std::int32_t{3}
        };
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{6},
            ormi::core::DType::float64
        );
        auto* input_data = static_cast<double*>(input->data());
        for (std::size_t i = 0; i < values.size(); ++i) {
            input_data[i] = values[i];
        }

        const auto output = ormi::core::ops::cast(input, ormi::core::DType::int32);
        const auto* output_data = static_cast<const std::int32_t*>(output->data());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(output_data[i], expected[i]);
        }
    }

    TEST(CPUCastTest, rejectsNonFiniteOrOutOfRangeInt32Conversion) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float64
        );
        auto* value = static_cast<double*>(input->data());

        *value = std::numeric_limits<double>::infinity();
        EXPECT_THROW(ormi::core::ops::cast(input, ormi::core::DType::int32)->realize(), ormi::DTypeError);

        *value = static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0;
        EXPECT_THROW(ormi::core::ops::cast(input, ormi::core::DType::int32)->realize(), ormi::DTypeError);
    }

    TEST(CPUCastTest, rejectsGradientThroughIntegerCast) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float32
        );
        const auto integer = ormi::core::ops::cast(input, ormi::core::DType::int32);
        const auto output = ormi::core::ops::cast(integer, ormi::core::DType::float32);

        EXPECT_THROW((void)ormi::core::grad(output, input), ormi::DTypeError);
    }

    TEST(CastTest, rejectsNullTensor) {
        EXPECT_THROW((void)ormi::core::ops::cast(nullptr, ormi::core::DType::float32), ormi::OperationError);
    }
}
