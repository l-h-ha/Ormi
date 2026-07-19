#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/attrs.h>
#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/optim/sgd.h>
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

    TEST(OptimizerTest, constructsOneLazySGDNode) {
        const auto parameter = floatTensor({3});
        const auto gradient = floatTensor({3});
        const auto next = ormi::core::optim::sgd(parameter, gradient, 0.1);

        EXPECT_FALSE(next->realized());
        EXPECT_EQ(next->shape(), parameter->shape());
        EXPECT_EQ(next->dtype(), parameter->dtype());
        ASSERT_NE(next->op(), nullptr);
        EXPECT_EQ(next->op()->name, "sgd");
        ASSERT_EQ(next->parents().size(), 2u);
        EXPECT_EQ(next->parents()[0], parameter);
        EXPECT_EQ(next->parents()[1], gradient);
        EXPECT_DOUBLE_EQ(std::get<ormi::core::SGD>(next->attrs()).step_size, 0.1);
    }

    TEST(OptimizerTest, zeroLearningRateReturnsParameter) {
        const auto parameter = floatTensor({1});
        const auto gradient = floatTensor({1});

        EXPECT_EQ(ormi::core::optim::sgd(parameter, gradient, 0.0), parameter);
    }

    TEST(CPUOptimizerTest, realizesWithoutMutatingParameter) {
        const auto parameter = floatTensor({3});
        const auto gradient = floatTensor({3});
        auto* parameter_data = static_cast<float*>(parameter->data());
        auto* gradient_data = static_cast<float*>(gradient->data());
        parameter_data[0] = 1.0f;
        parameter_data[1] = 2.0f;
        parameter_data[2] = 3.0f;
        gradient_data[0] = 0.5f;
        gradient_data[1] = 1.0f;
        gradient_data[2] = -2.0f;

        const auto next = ormi::core::optim::sgd(parameter, gradient, 0.1);
        const auto* next_data = static_cast<const float*>(next->data());

        EXPECT_FLOAT_EQ(next_data[0], 0.95f);
        EXPECT_FLOAT_EQ(next_data[1], 1.9f);
        EXPECT_FLOAT_EQ(next_data[2], 3.2f);
        EXPECT_FLOAT_EQ(parameter_data[0], 1.0f);
        EXPECT_FLOAT_EQ(parameter_data[1], 2.0f);
        EXPECT_FLOAT_EQ(parameter_data[2], 3.0f);
    }

    TEST(CPUOptimizerTest, supportsFloat64AndFloat16) {
        const auto parameter64 = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{1}, ormi::core::DType::float64
        );
        const auto gradient64 = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{1}, ormi::core::DType::float64
        );
        *static_cast<double*>(parameter64->data()) = 2.0;
        *static_cast<double*>(gradient64->data()) = 0.25;

        const auto parameter16 = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{1}, ormi::core::DType::float16
        );
        const auto gradient16 = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{1}, ormi::core::DType::float16
        );
        *static_cast<ormi::core::Float16*>(parameter16->data()) = ormi::core::Float16(2.0f);
        *static_cast<ormi::core::Float16*>(gradient16->data()) = ormi::core::Float16(0.25f);

        const auto next64 = ormi::core::optim::sgd(parameter64, gradient64, 0.5);
        const auto next16 = ormi::core::optim::sgd(parameter16, gradient16, 0.5);

        EXPECT_DOUBLE_EQ(*static_cast<const double*>(next64->data()), 1.875);
        EXPECT_FLOAT_EQ(
            static_cast<float>(*static_cast<const ormi::core::Float16*>(next16->data())),
            1.875f
        );
    }

    TEST(CPUOptimizerTest, readsTransposedParameterAndGradient) {
        const auto parameter = floatTensor({2, 2});
        const auto gradient = floatTensor({2, 2});
        auto* parameter_data = static_cast<float*>(parameter->data());
        auto* gradient_data = static_cast<float*>(gradient->data());

        for (std::size_t i = 0; i < 4; ++i) {
            parameter_data[i] = static_cast<float>(i + 1);
            gradient_data[i] = static_cast<float>((i + 1) * 10);
        }

        const auto parameter_view = ormi::core::ops::transpose(parameter, {1, 0});
        const auto gradient_view = ormi::core::ops::transpose(gradient, {1, 0});
        const auto next = ormi::core::optim::sgd(parameter_view, gradient_view, 0.01);
        const auto* data = static_cast<const float*>(next->data());

        EXPECT_FLOAT_EQ(data[0], 0.9f);
        EXPECT_FLOAT_EQ(data[1], 2.7f);
        EXPECT_FLOAT_EQ(data[2], 1.8f);
        EXPECT_FLOAT_EQ(data[3], 3.6f);
    }

    TEST(CPUOptimizerTest, updatesFloat32MasterFromFloat16Computation) {
        const auto parameter = floatTensor({});
        *static_cast<float*>(parameter->data()) = 3.0f;
        const auto half = ormi::core::ops::cast(parameter, ormi::core::DType::float16);
        const auto output = ormi::core::ops::mul(half, half);
        const auto gradient = ormi::core::grad(output, parameter);
        const auto next = ormi::core::optim::sgd(parameter, gradient, 0.1);

        EXPECT_EQ(gradient->dtype(), ormi::core::DType::float32);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(next->data()), 2.4f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(parameter->data()), 3.0f);
    }

    TEST(OptimizerTest, rejectsInvalidInputs) {
        const auto parameter = floatTensor({2});
        const auto wrong_shape = floatTensor({3});
        const auto wrong_dtype = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{2}, ormi::core::DType::float64
        );
        const auto integers = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{2}, ormi::core::DType::int32
        );

        EXPECT_THROW((void)ormi::core::optim::sgd(nullptr, parameter, 0.1), ormi::OperationError);
        EXPECT_THROW((void)ormi::core::optim::sgd(parameter, nullptr, 0.1), ormi::OperationError);
        EXPECT_THROW((void)ormi::core::optim::sgd(parameter, parameter, -0.1), ormi::OperationError);
        EXPECT_THROW(
            (void)ormi::core::optim::sgd(parameter, parameter, std::nan("")),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)ormi::core::optim::sgd(parameter, parameter, 0.1, 0.0),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)ormi::core::optim::sgd(parameter, parameter, 0.1, std::nan("")),
            ormi::OperationError
        );
        EXPECT_THROW(
            (void)ormi::core::optim::sgd(parameter, parameter, 1.0, 1e-320),
            ormi::OperationError
        );
        EXPECT_THROW((void)ormi::core::optim::sgd(parameter, wrong_shape, 0.1), ormi::ShapeError);
        EXPECT_THROW((void)ormi::core::optim::sgd(parameter, wrong_dtype, 0.1), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::optim::sgd(integers, integers, 0.1), ormi::DTypeError);
    }

    TEST(OptimizerTest, startsANewDifferentiableIteration) {
        const auto parameter = floatTensor({});
        const auto gradient = floatTensor({});
        *static_cast<float*>(parameter->data()) = 2.0f;
        *static_cast<float*>(gradient->data()) = 1.0f;
        const auto next = ormi::core::optim::sgd(parameter, gradient, 0.1);
        const auto next_loss = ormi::core::ops::mul(next, next);
        const auto next_gradient = ormi::core::grad(next_loss, next);
        const auto old_gradient = ormi::core::grad(next, parameter);

        EXPECT_FLOAT_EQ(*static_cast<const float*>(next_gradient->data()), 3.8f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(old_gradient->data()), 0.0f);
    }
}
