#include <array>
#include <cmath>
#include <memory>
#include <string_view>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/pointwise.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> floatTensor(ormi::core::Shape shape) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float32
        );
    }

    std::shared_ptr<ormi::core::Tensor> doubleTensor(ormi::core::Shape shape) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float64
        );
    }

    TEST(CPUPointwiseTest, reportsFloatingPointRequirementForTranscendentals) {
        const auto device = ormi::core::Device::cpu();
        ormi::core::Tensor input(device, {1}, ormi::core::DType::int32);
        ormi::core::Tensor output(device, {1}, ormi::core::DType::int32);
        const std::array operations{
            ormi::core::ops::UnaryOp::exp,
            ormi::core::ops::UnaryOp::log,
            ormi::core::ops::UnaryOp::sin,
            ormi::core::ops::UnaryOp::cos
        };

        for (const auto operation : operations) {
            try {
                ormi::backends::cpu::pointwise_unary(operation, output, input);
                FAIL() << "An integer transcendental operation must throw.";
            } catch (const ormi::DTypeError& error) {
                const std::string_view message(error.what());
                EXPECT_NE(message.find("requires a floating-point tensor"), std::string_view::npos);
            }
        }
    }

    TEST(PointwiseTest, addConstructsLazyTensor) {
        const auto lhs = floatTensor({2});
        const auto rhs = floatTensor({2});
        const auto result = ormi::core::ops::add(lhs, rhs);

        EXPECT_FALSE(result->realized());
        ASSERT_NE(result->op(), nullptr);
        EXPECT_EQ(result->op()->name, "add");
        ASSERT_EQ(result->parents().size(), 2u);
        EXPECT_EQ(result->parents()[0], lhs);
        EXPECT_EQ(result->parents()[1], rhs);
        EXPECT_EQ(result->shape(), (ormi::core::Shape{2}));
    }

    TEST(PointwiseTest, realizesAddOnce) {
        const auto lhs = floatTensor({3});
        const auto rhs = floatTensor({3});
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 1.0f;
        lhs_data[1] = 2.0f;
        lhs_data[2] = 3.0f;
        rhs_data[0] = 4.0f;
        rhs_data[1] = 5.0f;
        rhs_data[2] = 6.0f;

        const auto result = ormi::core::ops::add(lhs, rhs);
        result->realize();
        const auto first_buffer = result->buffer();
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_TRUE(result->realized());
        EXPECT_FLOAT_EQ(data[0], 5.0f);
        EXPECT_FLOAT_EQ(data[1], 7.0f);
        EXPECT_FLOAT_EQ(data[2], 9.0f);

        result->realize();
        EXPECT_EQ(result->buffer(), first_buffer);
    }

    TEST(PointwiseTest, realizesBroadcastAdd) {
        const auto lhs = floatTensor({2, 1});
        const auto rhs = floatTensor({1, 3});
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 10.0f;
        lhs_data[1] = 20.0f;
        rhs_data[0] = 1.0f;
        rhs_data[1] = 2.0f;
        rhs_data[2] = 3.0f;

        const auto result = ormi::core::ops::add(lhs, rhs);
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 3}));
        EXPECT_FLOAT_EQ(data[0], 11.0f);
        EXPECT_FLOAT_EQ(data[1], 12.0f);
        EXPECT_FLOAT_EQ(data[2], 13.0f);
        EXPECT_FLOAT_EQ(data[3], 21.0f);
        EXPECT_FLOAT_EQ(data[4], 22.0f);
        EXPECT_FLOAT_EQ(data[5], 23.0f);
    }

    TEST(PointwiseTest, realizesSubMulAndDiv) {
        const auto lhs = floatTensor({1});
        const auto rhs = floatTensor({1});
        *static_cast<float*>(lhs->data()) = 12.0f;
        *static_cast<float*>(rhs->data()) = 3.0f;

        const auto difference = ormi::core::ops::sub(lhs, rhs);
        const auto product = ormi::core::ops::mul(lhs, rhs);
        const auto quotient = ormi::core::ops::div(lhs, rhs);

        ASSERT_NE(difference->op(), nullptr);
        ASSERT_NE(product->op(), nullptr);
        ASSERT_NE(quotient->op(), nullptr);
        EXPECT_EQ(difference->op()->name, "sub");
        EXPECT_EQ(product->op()->name, "mul");
        EXPECT_EQ(quotient->op()->name, "div");
        EXPECT_FALSE(difference->realized());
        EXPECT_FALSE(product->realized());
        EXPECT_FALSE(quotient->realized());
        EXPECT_FLOAT_EQ(*static_cast<const float*>(difference->data()), 9.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(product->data()), 36.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(quotient->data()), 4.0f);
    }

    TEST(PointwiseTest, removesShapeSafeMultiplicativeIdentities) {
        const auto input = floatTensor({2, 3});
        const auto one = ormi::core::ops::ones_like(input);

        EXPECT_EQ(ormi::core::ops::mul(input, one), input);
        EXPECT_EQ(ormi::core::ops::mul(one, input), input);
        EXPECT_EQ(ormi::core::ops::div(input, one), input);
    }

    TEST(PointwiseTest, preservesBroadcastingWhenOneIsAnIdentity) {
        const auto input = floatTensor({2, 1});
        const auto identity_shape = floatTensor({1, 3});
        const auto one = ormi::core::ops::ones_like(identity_shape);
        const auto result = ormi::core::ops::mul(input, one);

        EXPECT_NE(result, input);
        EXPECT_NE(result, one);
        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 3}));
    }

    TEST(CPUPointwiseTest, realizesNeg) {
        const auto input = floatTensor({3});
        auto* input_data = static_cast<float*>(input->data());
        input_data[0] = -2.0f;
        input_data[1] = 0.0f;
        input_data[2] = 3.5f;

        const auto result = ormi::core::ops::neg(input);

        EXPECT_FALSE(result->realized());
        ASSERT_NE(result->op(), nullptr);
        EXPECT_EQ(result->op()->name, "neg");
        ASSERT_EQ(result->parents().size(), 1u);
        EXPECT_EQ(result->parents()[0], input);

        const auto* result_data = static_cast<const float*>(result->data());
        EXPECT_FLOAT_EQ(result_data[0], 2.0f);
        EXPECT_FLOAT_EQ(result_data[1], 0.0f);
        EXPECT_FLOAT_EQ(result_data[2], -3.5f);
    }

    TEST(CPUPointwiseTest, realizesNegOfTranspose) {
        const auto input = floatTensor({2, 3});
        auto* input_data = static_cast<float*>(input->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            input_data[i] = static_cast<float>(i + 1);
        }

        const auto transposed = ormi::core::ops::transpose(input, {1, 0});
        const auto result = ormi::core::ops::neg(transposed);
        const auto* result_data = static_cast<const float*>(result->data());

        EXPECT_EQ(result->shape(), (ormi::core::Shape{3, 2}));
        EXPECT_FLOAT_EQ(result_data[0], -1.0f);
        EXPECT_FLOAT_EQ(result_data[1], -4.0f);
        EXPECT_FLOAT_EQ(result_data[2], -2.0f);
        EXPECT_FLOAT_EQ(result_data[3], -5.0f);
        EXPECT_FLOAT_EQ(result_data[4], -3.0f);
        EXPECT_FLOAT_EQ(result_data[5], -6.0f);
    }

    TEST(CPUPointwiseTest, realizesNegOfExpandedTensor) {
        const auto input = floatTensor({2, 1});
        auto* input_data = static_cast<float*>(input->data());
        input_data[0] = 2.0f;
        input_data[1] = -3.0f;

        const auto expanded = ormi::core::ops::expand(input, {2, 3});
        const auto result = ormi::core::ops::neg(expanded);
        const auto* result_data = static_cast<const float*>(result->data());

        EXPECT_FLOAT_EQ(result_data[0], -2.0f);
        EXPECT_FLOAT_EQ(result_data[1], -2.0f);
        EXPECT_FLOAT_EQ(result_data[2], -2.0f);
        EXPECT_FLOAT_EQ(result_data[3], 3.0f);
        EXPECT_FLOAT_EQ(result_data[4], 3.0f);
        EXPECT_FLOAT_EQ(result_data[5], 3.0f);
    }

    TEST(CPUPointwiseTest, realizesExp) {
        const auto input = floatTensor({3});
        auto* input_data = static_cast<float*>(input->data());
        input_data[0] = -1.0f;
        input_data[1] = 0.0f;
        input_data[2] = 1.0f;

        const auto result = ormi::core::ops::exp(input);

        EXPECT_FALSE(result->realized());
        ASSERT_NE(result->op(), nullptr);
        EXPECT_EQ(result->op()->name, "exp");

        const auto* result_data = static_cast<const float*>(result->data());
        EXPECT_FLOAT_EQ(result_data[0], std::exp(-1.0f));
        EXPECT_FLOAT_EQ(result_data[1], 1.0f);
        EXPECT_FLOAT_EQ(result_data[2], std::exp(1.0f));
    }

    TEST(PointwiseTest, rejectsInt32TranscendentalAtConstruction) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::int32
        );

        EXPECT_THROW((void)ormi::core::ops::exp(input), ormi::DTypeError);
    }

    TEST(CPUPointwiseTest, realizesLogSinAndCos) {
        const auto input = floatTensor({2});
        auto* input_data = static_cast<float*>(input->data());
        input_data[0] = 0.5f;
        input_data[1] = 2.0f;

        const auto logarithm = ormi::core::ops::log(input);
        const auto sine = ormi::core::ops::sin(input);
        const auto cosine = ormi::core::ops::cos(input);
        const auto* log_data = static_cast<const float*>(logarithm->data());
        const auto* sin_data = static_cast<const float*>(sine->data());
        const auto* cos_data = static_cast<const float*>(cosine->data());

        EXPECT_FLOAT_EQ(log_data[0], std::log(0.5f));
        EXPECT_FLOAT_EQ(log_data[1], std::log(2.0f));
        EXPECT_FLOAT_EQ(sin_data[0], std::sin(0.5f));
        EXPECT_FLOAT_EQ(sin_data[1], std::sin(2.0f));
        EXPECT_FLOAT_EQ(cos_data[0], std::cos(0.5f));
        EXPECT_FLOAT_EQ(cos_data[1], std::cos(2.0f));
    }

    TEST(CPUPointwiseTest, realizesFloat64ArithmeticAndUnaryOperations) {
        const auto lhs = doubleTensor({2});
        const auto rhs = doubleTensor({2});
        auto* lhs_data = static_cast<double*>(lhs->data());
        auto* rhs_data = static_cast<double*>(rhs->data());
        lhs_data[0] = 0.5;
        lhs_data[1] = 1.5;
        rhs_data[0] = 2.0;
        rhs_data[1] = 3.0;

        const auto sum = ormi::core::ops::add(lhs, rhs);
        const auto exponential = ormi::core::ops::exp(lhs);
        const auto* sum_data = static_cast<const double*>(sum->data());
        const auto* exp_data = static_cast<const double*>(exponential->data());

        EXPECT_DOUBLE_EQ(sum_data[0], 2.5);
        EXPECT_DOUBLE_EQ(sum_data[1], 4.5);
        EXPECT_DOUBLE_EQ(exp_data[0], std::exp(0.5));
        EXPECT_DOUBLE_EQ(exp_data[1], std::exp(1.5));
    }

    TEST(PointwiseTest, realizesGraphRootWithoutMaterializingFusedParent) {
        const auto lhs = floatTensor({1});
        const auto rhs = floatTensor({1});
        *static_cast<float*>(lhs->data()) = 2.0f;
        *static_cast<float*>(rhs->data()) = 3.0f;
        const auto parent = ormi::core::ops::add(lhs, rhs);
        const auto result = ormi::core::ops::add(parent, rhs);

        EXPECT_FALSE(parent->realized());
        EXPECT_FALSE(result->realized());

        const auto* data = static_cast<const float*>(result->data());

        EXPECT_FALSE(parent->realized());
        EXPECT_TRUE(result->realized());
        EXPECT_FLOAT_EQ(data[0], 8.0f);
    }

    TEST(CPUPointwiseTest, realizesFloat16ArithmeticAndUnaryOperations) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float16
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float16
        );
        auto* lhs_data = static_cast<ormi::core::Float16*>(lhs->data());
        auto* rhs_data = static_cast<ormi::core::Float16*>(rhs->data());
        lhs_data[0] = ormi::core::Float16(1.5f);
        rhs_data[0] = ormi::core::Float16(0.5f);
        const auto result = ormi::core::ops::add(lhs, rhs);
        const auto exponential = ormi::core::ops::exp(rhs);

        EXPECT_FLOAT_EQ(static_cast<float>(*static_cast<const ormi::core::Float16*>(result->data())), 2.0f);
        EXPECT_NEAR(
            static_cast<float>(*static_cast<const ormi::core::Float16*>(exponential->data())),
            std::exp(0.5f),
            0.001f
        );
    }

    TEST(CPUPointwiseTest, promotesFloatingInputsWithOneLazyCast) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float16
        );
        const auto rhs = floatTensor({2});
        auto* lhs_data = static_cast<ormi::core::Float16*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = ormi::core::Float16(1.5f);
        lhs_data[1] = ormi::core::Float16(2.0f);
        rhs_data[0] = 0.5f;
        rhs_data[1] = 3.0f;

        const auto result = ormi::core::ops::add(lhs, rhs);

        EXPECT_EQ(result->dtype(), ormi::core::DType::float32);
        ASSERT_EQ(result->parents().size(), 2u);
        ASSERT_NE(result->parents()[0]->op(), nullptr);
        EXPECT_EQ(result->parents()[0]->op()->name, "cast");
        EXPECT_EQ(result->parents()[0]->parents()[0], lhs);
        EXPECT_EQ(result->parents()[1], rhs);
        const auto* data = static_cast<const float*>(result->data());
        EXPECT_FLOAT_EQ(data[0], 2.0f);
        EXPECT_FLOAT_EQ(data[1], 5.0f);
    }

    TEST(PointwiseTest, removesMixedDtypeMultiplicativeIdentity) {
        const auto input = floatTensor({2});
        const auto half = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float16
        );
        const auto one = ormi::core::ops::ones_like(half);

        EXPECT_EQ(ormi::core::ops::mul(one, input), input);
    }

    TEST(PointwiseTest, rejectsImplicitIntegerFloatingPromotion) {
        const auto lhs = floatTensor({2});
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::int32
        );

        EXPECT_THROW((void)ormi::core::ops::add(lhs, rhs), ormi::DTypeError);
    }

    TEST(CPUPointwiseTest, realizesInt32Negation) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::int32
        );
        auto* values = static_cast<std::int32_t*>(input->data());
        values[0] = -2;
        values[1] = 3;

        const auto result = ormi::core::ops::neg(input);
        const auto* data = static_cast<const std::int32_t*>(result->data());

        EXPECT_EQ(data[0], 2);
        EXPECT_EQ(data[1], -3);
    }

    TEST(PointwiseTest, rejectsBooleanArithmeticAtConstruction) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::boolean
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::boolean
        );

        EXPECT_THROW((void)ormi::core::ops::add(lhs, rhs), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::ops::sub(lhs, rhs), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::ops::mul(lhs, rhs), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::ops::div(lhs, rhs), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::ops::neg(lhs), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::ops::exp(lhs), ormi::DTypeError);
    }

    TEST(PointwiseTest, rejectsNullTensor) {
        const auto rhs = floatTensor({2});

        EXPECT_THROW((void)ormi::core::ops::add(nullptr, rhs), ormi::OperationError);
        EXPECT_THROW((void)ormi::core::ops::neg(nullptr), ormi::OperationError);
    }
}
