#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/attrs.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> floatTensor(ormi::core::Shape shape) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float32
        );
    }

    TEST(ReductionTest, sumConstructsLazyTensorWithNormalizedAxes) {
        const auto input = floatTensor({2, 3, 4});
        const auto result = ormi::core::ops::sum(
            input,
            std::vector<std::int64_t>{-1, 0},
            true
        );
        const auto& attrs = std::get<ormi::core::Reduce>(result->attrs());

        EXPECT_FALSE(result->realized());
        ASSERT_NE(result->op(), nullptr);
        EXPECT_EQ(result->op()->name, "sum");
        EXPECT_EQ(result->shape(), (ormi::core::Shape{1, 3, 1}));
        EXPECT_EQ(attrs.axes, (std::vector<std::int64_t>{0, 2}));
        EXPECT_TRUE(attrs.keepdims);
        ASSERT_EQ(result->parents().size(), 1u);
        EXPECT_EQ(result->parents()[0], input);
    }

    TEST(ReductionTest, realizesGlobalSum) {
        const auto input = floatTensor({2, 3});
        auto* input_data = static_cast<float*>(input->data());

        for (std::size_t i = 0; i < 6; ++i) {
            input_data[i] = static_cast<float>(i + 1);
        }

        const auto result = ormi::core::ops::sum(input);
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_TRUE(result->shape().scalar());
        EXPECT_FLOAT_EQ(data[0], 21.0f);
    }

    TEST(ReductionTest, realizesMultipleAxesWithKeepdims) {
        const auto input = floatTensor({2, 2, 2});
        auto* input_data = static_cast<float*>(input->data());

        for (std::size_t i = 0; i < 8; ++i) {
            input_data[i] = static_cast<float>(i + 1);
        }

        const auto result = ormi::core::ops::sum(
            input,
            std::vector<std::int64_t>{0, 2},
            true
        );
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_EQ(result->shape(), (ormi::core::Shape{1, 2, 1}));
        EXPECT_FLOAT_EQ(data[0], 14.0f);
        EXPECT_FLOAT_EQ(data[1], 22.0f);
    }

    TEST(ReductionTest, singleAxisListDoesNotBindKeepdimsOverload) {
        const auto input = floatTensor({2, 3});
        const auto result = ormi::core::ops::sum(input, {1});
        const auto& attrs = std::get<ormi::core::Reduce>(result->attrs());

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2}));
        EXPECT_EQ(attrs.axes, (std::vector<std::int64_t>{1}));
        EXPECT_FALSE(attrs.keepdims);
    }

    TEST(ReductionTest, realizesFloat64Sum) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float64
        );
        auto* input_data = static_cast<double*>(input->data());
        input_data[0] = 0.25;
        input_data[1] = 0.5;
        input_data[2] = 0.75;
        input_data[3] = 1.0;

        const auto result = ormi::core::ops::sum(input);

        EXPECT_DOUBLE_EQ(*static_cast<const double*>(result->data()), 2.5);
    }

    TEST(CPUReductionTest, realizesFloat16SumWithFloat32Accumulation) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float16
        );
        auto* input_data = static_cast<ormi::core::Float16*>(input->data());
        input_data[0] = ormi::core::Float16(2048.0f);
        input_data[1] = ormi::core::Float16(1.0f);
        input_data[2] = ormi::core::Float16(1.0f);
        input_data[3] = ormi::core::Float16(-2048.0f);

        const auto result = ormi::core::ops::sum(input);
        const auto value = *static_cast<const ormi::core::Float16*>(result->data());

        EXPECT_FLOAT_EQ(static_cast<float>(value), 2.0f);
    }

    TEST(ReductionTest, rejectsInvalidAxes) {
        const auto input = floatTensor({2, 3});

        EXPECT_THROW(
            (void)ormi::core::ops::sum(input, std::vector<std::int64_t>{0, 0}),
            ormi::ShapeError
        );
        EXPECT_THROW(
            (void)ormi::core::ops::sum(input, std::vector<std::int64_t>{2}),
            ormi::ShapeError
        );
        EXPECT_THROW((void)ormi::core::ops::sum(nullptr), ormi::OperationError);
    }

    TEST(ReductionTest, rejectsBooleanSumAtConstruction) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::boolean
        );

        EXPECT_THROW((void)ormi::core::ops::sum(input), ormi::DTypeError);
    }
}
