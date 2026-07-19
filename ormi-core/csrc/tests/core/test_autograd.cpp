#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
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

    std::shared_ptr<ormi::core::Tensor> doubleTensor(ormi::core::Shape shape) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float64
        );
    }

    TEST(AutogradTest, differentiatesSquareLazily) {
        const auto x = floatTensor({});
        *static_cast<float*>(x->data()) = 3.0f;
        const auto y = ormi::core::ops::mul(x, x);
        const auto dx = ormi::core::grad(y, x);

        EXPECT_FALSE(dx->realized());
        EXPECT_FLOAT_EQ(*static_cast<const float*>(dx->data()), 6.0f);
    }

    TEST(CPUAutogradTest, differentiatesFloat16SquareTwiceLazily) {
        const auto x = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float16
        );
        *static_cast<ormi::core::Float16*>(x->data()) = ormi::core::Float16(3.0f);
        const auto y = ormi::core::ops::mul(x, x);
        const auto dx = ormi::core::grad(y, x);
        const auto ddx = ormi::core::grad(dx, x);

        EXPECT_FALSE(dx->realized());
        EXPECT_FALSE(ddx->realized());
        EXPECT_FLOAT_EQ(
            static_cast<float>(*static_cast<const ormi::core::Float16*>(dx->data())),
            6.0f
        );
        EXPECT_FLOAT_EQ(
            static_cast<float>(*static_cast<const ormi::core::Float16*>(ddx->data())),
            2.0f
        );
    }

    TEST(AutogradTest, reversesBroadcasting) {
        const auto lhs = floatTensor({2, 1});
        const auto rhs = floatTensor({1, 3});
        const auto output = ormi::core::ops::add(lhs, rhs);
        const auto seed = ormi::core::ops::ones_like(output);
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs},
            seed
        );
        const auto* lhs_gradient = static_cast<const float*>(gradients[0]->data());
        const auto* rhs_gradient = static_cast<const float*>(gradients[1]->data());

        EXPECT_FLOAT_EQ(lhs_gradient[0], 3.0f);
        EXPECT_FLOAT_EQ(lhs_gradient[1], 3.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[0], 2.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[1], 2.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[2], 2.0f);
    }

    TEST(AutogradTest, differentiatesDivision) {
        const auto lhs = floatTensor({});
        const auto rhs = floatTensor({});
        *static_cast<float*>(lhs->data()) = 6.0f;
        *static_cast<float*>(rhs->data()) = 2.0f;
        const auto output = ormi::core::ops::div(lhs, rhs);
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );

        EXPECT_FLOAT_EQ(*static_cast<const float*>(gradients[0]->data()), 0.5f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(gradients[1]->data()), -1.5f);
    }

    TEST(AutogradTest, differentiatesNeg) {
        const auto x = floatTensor({});
        *static_cast<float*>(x->data()) = 4.0f;
        const auto y = ormi::core::ops::neg(x);
        const auto dx = ormi::core::grad(y, x);

        EXPECT_FALSE(dx->realized());
        EXPECT_FLOAT_EQ(*static_cast<const float*>(dx->data()), -1.0f);
    }

    TEST(AutogradTest, differentiatesExpByReusingItsForwardNode) {
        const auto x = floatTensor({});
        *static_cast<float*>(x->data()) = 1.0f;
        const auto y = ormi::core::ops::exp(x);
        const auto dx = ormi::core::grad(y, x);
        const auto ddx = ormi::core::grad(dx, x);

        EXPECT_EQ(dx, y);
        EXPECT_EQ(ddx, y);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(ddx->data()), std::exp(1.0f));
    }

    TEST(AutogradTest, differentiatesLogSinAndCos) {
        const auto x = floatTensor({});
        *static_cast<float*>(x->data()) = 0.5f;
        const auto log_gradient = ormi::core::grad(ormi::core::ops::log(x), x);
        const auto sin_gradient = ormi::core::grad(ormi::core::ops::sin(x), x);
        const auto cos_gradient = ormi::core::grad(ormi::core::ops::cos(x), x);

        EXPECT_FLOAT_EQ(*static_cast<const float*>(log_gradient->data()), 2.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(sin_gradient->data()), std::cos(0.5f));
        EXPECT_FLOAT_EQ(*static_cast<const float*>(cos_gradient->data()), -std::sin(0.5f));
    }

    TEST(AutogradTest, supportsFifthOrderDifferentiation) {
        const auto x = floatTensor({});
        *static_cast<float*>(x->data()) = 2.0f;
        const auto x2 = ormi::core::ops::mul(x, x);
        const auto x3 = ormi::core::ops::mul(x2, x);
        const auto x4 = ormi::core::ops::mul(x3, x);
        const auto x5 = ormi::core::ops::mul(x4, x);
        const auto fifth = ormi::core::diff(x5, x, 5);
        const auto sixth = ormi::core::diff(x5, x, 6);

        EXPECT_FLOAT_EQ(*static_cast<const float*>(fifth->data()), 120.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(sixth->data()), 0.0f);
    }

    TEST(AutogradTest, supportsFloat64HigherOrderDifferentiation) {
        const auto x = doubleTensor({});
        *static_cast<double*>(x->data()) = 2.0;
        const auto y = ormi::core::ops::mul(x, x);
        const auto first = ormi::core::grad(y, x);
        const auto second = ormi::core::grad(first, x);

        EXPECT_DOUBLE_EQ(*static_cast<const double*>(first->data()), 4.0);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(second->data()), 2.0);
    }

    TEST(AutogradTest, requiresSeedForNonScalarOutput) {
        const auto x = floatTensor({2});
        const auto y = ormi::core::ops::mul(x, x);

        EXPECT_THROW((void)ormi::core::grad(y, x), ormi::ShapeError);
    }

    TEST(AutogradTest, rejectsNonFloatingTensors) {
        const auto integer = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::int32
        );
        const auto boolean = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::boolean
        );

        EXPECT_THROW((void)ormi::core::grad(integer, integer), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::grad(boolean, boolean), ormi::DTypeError);
    }
}
