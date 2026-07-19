#include <cmath>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/attrs.h>
#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/optim/grad_scaler.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> floatScalar(float value) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float32
        );
        *static_cast<float*>(tensor->data()) = value;
        return tensor;
    }

    TEST(GradScalerTest, requiresFinitePositiveScale) {
        EXPECT_THROW((void)ormi::core::optim::GradScaler(0.0), ormi::OperationError);
        EXPECT_THROW((void)ormi::core::optim::GradScaler(0.5), ormi::OperationError);
        EXPECT_THROW((void)ormi::core::optim::GradScaler(-1.0), ormi::OperationError);
        EXPECT_THROW((void)ormi::core::optim::GradScaler(std::nan("")), ormi::OperationError);
        EXPECT_THROW(
            (void)ormi::core::optim::GradScaler(std::numeric_limits<double>::infinity()),
            ormi::OperationError
        );
    }

    TEST(GradScalerTest, validatesScalePolicy) {
        using ormi::core::optim::GradScaler;
        using ormi::core::optim::ScalePolicy;

        EXPECT_THROW(
            (void)GradScaler(8.0, ScalePolicy{.growth = 1.0}),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)GradScaler(8.0, ScalePolicy{.backoff = 0.0}),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)GradScaler(8.0, ScalePolicy{.interval = 0}),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)GradScaler(8.0, ScalePolicy{.limit = 4.0}),
            ormi::OperationError
        );
    }

    TEST(GradScalerTest, scaleOneReturnsLoss) {
        const auto loss = floatScalar(2.0f);
        const ormi::core::optim::GradScaler scaler(1.0);

        EXPECT_EQ(scaler.value(), 1.0);
        EXPECT_EQ(scaler.scale(loss), loss);
    }

    TEST(GradScalerTest, constructsOneArithmeticScaleNode) {
        const auto loss = floatScalar(2.0f);
        const ormi::core::optim::GradScaler scaler(8.0);
        const auto scaled = scaler.scale(loss);

        EXPECT_FALSE(scaled->realized());
        ASSERT_NE(scaled->op(), nullptr);
        EXPECT_EQ(scaled->op()->name, "mul");
        ASSERT_EQ(scaled->parents().size(), 2u);
        EXPECT_EQ(scaled->parents()[0], loss);

        const auto& factor = scaled->parents()[1];
        EXPECT_FALSE(factor->realized());
        EXPECT_TRUE(factor->shape().scalar());
        EXPECT_EQ(factor->dtype(), loss->dtype());
        ASSERT_NE(factor->op(), nullptr);
        EXPECT_EQ(factor->op()->name, "full");

        EXPECT_FLOAT_EQ(*static_cast<const float*>(scaled->data()), 16.0f);
    }

    TEST(GradScalerTest, rejectsUnsupportedLosses) {
        const ormi::core::optim::GradScaler scaler(8.0);
        const auto integer = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{}, ormi::core::DType::int32
        );
        const auto half = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{}, ormi::core::DType::float16
        );
        const ormi::core::optim::GradScaler too_large_for_half(65536.0);

        EXPECT_THROW((void)scaler.scale(nullptr), ormi::OperationError);
        EXPECT_THROW((void)scaler.scale(integer), ormi::DTypeError);
        EXPECT_THROW((void)too_large_for_half.scale(half), ormi::DTypeError);
    }

    TEST(CPUGradScalerTest, fusesUnscaleIntoSGD) {
        const auto parameter = floatScalar(3.0f);
        const auto scaled_gradient = floatScalar(48.0f);
        ormi::core::optim::GradScaler scaler(8.0);
        const auto next = scaler.step(parameter, scaled_gradient, 0.1);

        ASSERT_TRUE(next.has_value());
        EXPECT_FALSE((*next)->realized());
        ASSERT_NE((*next)->op(), nullptr);
        EXPECT_EQ((*next)->op()->name, "sgd");
        ASSERT_EQ((*next)->parents().size(), 2u);
        EXPECT_EQ((*next)->parents()[0], parameter);
        EXPECT_EQ((*next)->parents()[1], scaled_gradient);

        const auto& options = std::get<ormi::core::SGD>((*next)->attrs());
        EXPECT_DOUBLE_EQ(options.step_size, 0.0125);
        EXPECT_FLOAT_EQ(*static_cast<const float*>((*next)->data()), 2.4f);
    }

    TEST(CPUGradScalerTest, scalesBackwardAndUnscalesUpdate) {
        const auto parameter = floatScalar(3.0f);
        const auto loss = ormi::core::ops::mul(parameter, parameter);
        ormi::core::optim::GradScaler scaler(8.0);
        const auto scaled_loss = scaler.scale(loss);
        const auto scaled_gradient = ormi::core::grad(scaled_loss, parameter);
        const auto next = scaler.step(parameter, scaled_gradient, 0.1);

        ASSERT_TRUE(next.has_value());
        EXPECT_FLOAT_EQ(*static_cast<const float*>(scaled_gradient->data()), 48.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>((*next)->data()), 2.4f);
    }

    TEST(CPUGradScalerTest, singleStepSkipsUpdateAndBacksOffOnOverflow) {
        const auto parameter = floatScalar(3.0f);
        const auto gradient = floatScalar(std::numeric_limits<float>::infinity());
        ormi::core::optim::GradScaler scaler(8.0);
        const auto next = scaler.step(parameter, gradient, 0.1);

        EXPECT_FALSE(next.has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 4.0);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(parameter->data()), 3.0f);
    }

    TEST(CPUGradScalerTest, batchStepUsesCapturedScaleBeforeGrowth) {
        const auto parameter0 = floatScalar(3.0f);
        const auto parameter1 = floatScalar(5.0f);
        const auto gradient0 = floatScalar(48.0f);
        const auto gradient1 = floatScalar(16.0f);
        ormi::core::optim::GradScaler scaler(
            8.0,
            ormi::core::optim::ScalePolicy{.interval = 1}
        );
        const auto result = scaler.step(
            std::vector<std::shared_ptr<ormi::core::Tensor>>{parameter0, parameter1},
            std::vector<std::shared_ptr<ormi::core::Tensor>>{gradient0, gradient1},
            0.1
        );

        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result->size(), 2u);
        EXPECT_DOUBLE_EQ(scaler.value(), 16.0);
        EXPECT_FALSE((*result)[0]->realized());
        EXPECT_FALSE((*result)[1]->realized());
        EXPECT_DOUBLE_EQ(std::get<ormi::core::SGD>((*result)[0]->attrs()).step_size, 0.0125);
        EXPECT_DOUBLE_EQ(std::get<ormi::core::SGD>((*result)[1]->attrs()).step_size, 0.0125);
        EXPECT_FLOAT_EQ(*static_cast<const float*>((*result)[0]->data()), 2.4f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>((*result)[1]->data()), 4.8f);
    }

    TEST(CPUGradScalerTest, batchStepSkipsAllUpdatesAndBacksOffOnOverflow) {
        const auto parameter0 = floatScalar(3.0f);
        const auto parameter1 = floatScalar(5.0f);
        const auto gradient0 = floatScalar(48.0f);
        const auto gradient1 = floatScalar(std::numeric_limits<float>::infinity());
        ormi::core::optim::GradScaler scaler(8.0);
        const auto result = scaler.step(
            std::vector<std::shared_ptr<ormi::core::Tensor>>{parameter0, parameter1},
            std::vector<std::shared_ptr<ormi::core::Tensor>>{gradient0, gradient1},
            0.1
        );

        EXPECT_FALSE(result.has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 4.0);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(parameter0->data()), 3.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(parameter1->data()), 5.0f);
    }

    TEST(CPUGradScalerTest, batchStepTracksConsecutiveFiniteSteps) {
        const auto parameter = floatScalar(3.0f);
        const auto gradient = floatScalar(48.0f);
        ormi::core::optim::GradScaler scaler(
            8.0,
            ormi::core::optim::ScalePolicy{.growth = 4.0, .interval = 2}
        );
        const std::vector<std::shared_ptr<ormi::core::Tensor>> parameters{parameter};
        const std::vector<std::shared_ptr<ormi::core::Tensor>> gradients{gradient};

        ASSERT_TRUE(scaler.step(parameters, gradients, 0.1).has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 8.0);
        ASSERT_TRUE(scaler.step(parameters, gradients, 0.1).has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 32.0);

        *static_cast<float*>(gradient->data()) = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(scaler.step(parameters, gradients, 0.1).has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 16.0);
    }

    TEST(CPUGradScalerTest, growthSaturatesAtPolicyLimit) {
        const auto parameter = floatScalar(3.0f);
        const auto gradient = floatScalar(48.0f);
        ormi::core::optim::GradScaler scaler(
            8.0,
            ormi::core::optim::ScalePolicy{
                .growth = 4.0,
                .interval = 1,
                .limit = 16.0
            }
        );

        ASSERT_TRUE(scaler.step(parameter, gradient, 0.1).has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 16.0);
        ASSERT_TRUE(scaler.step(parameter, gradient, 0.1).has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 16.0);
    }

    TEST(CPUGradScalerTest, batchStepNeverBacksScaleBelowOne) {
        const auto parameter = floatScalar(3.0f);
        const auto gradient = floatScalar(std::numeric_limits<float>::infinity());
        ormi::core::optim::GradScaler scaler(1.0);

        EXPECT_FALSE(
            scaler.step(
                std::vector<std::shared_ptr<ormi::core::Tensor>>{parameter},
                std::vector<std::shared_ptr<ormi::core::Tensor>>{gradient},
                0.1
            ).has_value()
        );
        EXPECT_DOUBLE_EQ(scaler.value(), 1.0);
    }

    TEST(GradScalerTest, batchStepRejectsInvalidTransactionsWithoutChangingScale) {
        const auto parameter = floatScalar(3.0f);
        const auto gradient = floatScalar(48.0f);
        const auto wrong_shape = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{2}, ormi::core::DType::float32
        );
        ormi::core::optim::GradScaler scaler(8.0);

        EXPECT_THROW(
            (void)scaler.step(
                std::vector<std::shared_ptr<ormi::core::Tensor>>{},
                std::vector<std::shared_ptr<ormi::core::Tensor>>{},
                0.1
            ),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)scaler.step(
                std::vector<std::shared_ptr<ormi::core::Tensor>>{parameter},
                std::vector<std::shared_ptr<ormi::core::Tensor>>{gradient, gradient},
                0.1
            ),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)scaler.step(
                std::vector<std::shared_ptr<ormi::core::Tensor>>{parameter, parameter},
                std::vector<std::shared_ptr<ormi::core::Tensor>>{gradient, wrong_shape},
                0.1
            ),
            ormi::ShapeError
        );
        EXPECT_DOUBLE_EQ(scaler.value(), 8.0);
    }

    TEST(GradScalerTest, validatesTransactionBeforeCheckingOverflow) {
        const auto parameter = floatScalar(3.0f);
        const auto wrong_shape = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{2}, ormi::core::DType::float32
        );
        auto* gradient = static_cast<float*>(wrong_shape->data());
        gradient[0] = std::numeric_limits<float>::infinity();
        gradient[1] = 1.0f;
        ormi::core::optim::GradScaler scaler(8.0);

        EXPECT_THROW(
            (void)scaler.step(parameter, wrong_shape, 0.1),
            ormi::ShapeError
        );
        EXPECT_DOUBLE_EQ(scaler.value(), 8.0);
    }
}
