#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autocast.h>
#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/optim/grad_scaler.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> tensor(
        ormi::core::Shape shape,
        ormi::core::DType dtype
    ) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            dtype
        );
    }

    TEST(CPUAutocastTest, lowersMatmulWithinScopeAndRestoresAfterward) {
        const auto lhs = tensor({2, 2}, ormi::core::DType::float32);
        const auto rhs = tensor({2, 2}, ormi::core::DType::float32);

        {
            const ormi::core::Autocast autocast;
            const auto result = ormi::core::ops::matmul(lhs, rhs);

            EXPECT_EQ(result->dtype(), ormi::core::DType::float16);
            ASSERT_NE(result->parents()[0]->op(), nullptr);
            ASSERT_NE(result->parents()[1]->op(), nullptr);
            EXPECT_EQ(result->parents()[0]->op()->name, "cast");
            EXPECT_EQ(result->parents()[1]->op()->name, "cast");

            {
                const ormi::core::Autocast nested;
            }

            EXPECT_EQ(
                ormi::core::ops::matmul(lhs, rhs)->dtype(),
                ormi::core::DType::float16
            );
        }

        const auto restored = ormi::core::ops::matmul(lhs, rhs);
        EXPECT_EQ(restored->dtype(), ormi::core::DType::float32);
        EXPECT_EQ(restored->parents()[0], lhs);
        EXPECT_EQ(restored->parents()[1], rhs);
    }

    TEST(CPUAutocastTest, lowersMixedMatmulWithoutRoundTripCast) {
        const auto lhs = tensor({1, 2}, ormi::core::DType::float16);
        const auto rhs = tensor({2, 1}, ormi::core::DType::float32);
        const ormi::core::Autocast autocast;
        const auto result = ormi::core::ops::matmul(lhs, rhs);

        EXPECT_EQ(result->dtype(), ormi::core::DType::float16);
        EXPECT_EQ(result->parents()[0], lhs);
        ASSERT_NE(result->parents()[1]->op(), nullptr);
        EXPECT_EQ(result->parents()[1]->op()->name, "cast");
        EXPECT_EQ(result->parents()[1]->parents()[0], rhs);
    }

    TEST(CPUAutocastTest, keepsFloat64MatmulAtFloat64) {
        const auto lhs = tensor({1, 1}, ormi::core::DType::float64);
        const auto rhs = tensor({1, 1}, ormi::core::DType::float64);
        const ormi::core::Autocast autocast;
        const auto result = ormi::core::ops::matmul(lhs, rhs);

        EXPECT_EQ(result->dtype(), ormi::core::DType::float64);
        EXPECT_EQ(result->parents()[0], lhs);
        EXPECT_EQ(result->parents()[1], rhs);
    }

    TEST(CPUAutocastTest, widensSensitiveFloat16Operations) {
        const auto input = tensor({2}, ormi::core::DType::float16);

        {
            const ormi::core::Autocast autocast;
            const auto exponential = ormi::core::ops::exp(input);
            const auto total = ormi::core::ops::sum(input);

            EXPECT_EQ(exponential->dtype(), ormi::core::DType::float32);
            EXPECT_EQ(total->dtype(), ormi::core::DType::float32);
            ASSERT_NE(exponential->parents()[0]->op(), nullptr);
            ASSERT_NE(total->parents()[0]->op(), nullptr);
            EXPECT_EQ(exponential->parents()[0]->op()->name, "cast");
            EXPECT_EQ(total->parents()[0]->op()->name, "cast");
        }

        EXPECT_EQ(ormi::core::ops::exp(input)->dtype(), ormi::core::DType::float16);
        EXPECT_EQ(ormi::core::ops::sum(input)->dtype(), ormi::core::DType::float16);
    }

    TEST(CPUAutocastTest, preservesPointwiseAndExplicitCastDTypes) {
        const auto lhs = tensor({2}, ormi::core::DType::float32);
        const auto rhs = tensor({2}, ormi::core::DType::float32);
        const ormi::core::Autocast autocast;
        const auto result = ormi::core::ops::add(lhs, rhs);
        const auto explicit_cast = ormi::core::ops::cast(lhs, ormi::core::DType::float64);

        EXPECT_EQ(result->dtype(), ormi::core::DType::float32);
        EXPECT_EQ(result->parents()[0], lhs);
        EXPECT_EQ(result->parents()[1], rhs);
        EXPECT_EQ(explicit_cast->dtype(), ormi::core::DType::float64);
    }

    TEST(CPUAutocastTest, returnsGradientsToFloat32MasterTensors) {
        const auto lhs = tensor({1, 2}, ormi::core::DType::float32);
        const auto rhs = tensor({2, 1}, ormi::core::DType::float32);
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 1.0f;
        lhs_data[1] = 2.0f;
        rhs_data[0] = 3.0f;
        rhs_data[1] = 4.0f;
        std::shared_ptr<ormi::core::Tensor> loss;

        {
            const ormi::core::Autocast autocast;
            loss = ormi::core::ops::sum(ormi::core::ops::matmul(lhs, rhs));
        }

        const auto gradients = ormi::core::grad(
            loss,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );
        const auto* lhs_gradient = static_cast<const float*>(gradients[0]->data());
        const auto* rhs_gradient = static_cast<const float*>(gradients[1]->data());

        EXPECT_EQ(gradients[0]->dtype(), ormi::core::DType::float32);
        EXPECT_EQ(gradients[1]->dtype(), ormi::core::DType::float32);
        EXPECT_FLOAT_EQ(lhs_gradient[0], 3.0f);
        EXPECT_FLOAT_EQ(lhs_gradient[1], 4.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[0], 1.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[1], 2.0f);
    }

    TEST(CPUAutocastTest, keepsPolicyThreadLocal) {
        const auto lhs = tensor({1, 1}, ormi::core::DType::float32);
        const auto rhs = tensor({1, 1}, ormi::core::DType::float32);
        auto worker_dtype = ormi::core::DType::boolean;

        {
            const ormi::core::Autocast autocast;
            std::thread worker([&] {
                worker_dtype = ormi::core::ops::matmul(lhs, rhs)->dtype();
            });
            worker.join();

            EXPECT_EQ(
                ormi::core::ops::matmul(lhs, rhs)->dtype(),
                ormi::core::DType::float16
            );
        }

        EXPECT_EQ(worker_dtype, ormi::core::DType::float32);
    }

    TEST(CPUAutocastTest, integratesWithScaledBackwardAndFloat32MasterUpdate) {
        const auto parameter = tensor({1, 1}, ormi::core::DType::float32);
        const auto input = tensor({1, 1}, ormi::core::DType::float32);
        *static_cast<float*>(parameter->data()) = 2.0f;
        *static_cast<float*>(input->data()) = 3.0f;
        ormi::core::optim::GradScaler scaler(8.0);
        std::shared_ptr<ormi::core::Tensor> loss;

        {
            const ormi::core::Autocast autocast;
            loss = ormi::core::ops::sum(ormi::core::ops::matmul(parameter, input));
        }

        const auto scaled_loss = scaler.scale(loss);
        const auto scaled_gradient = ormi::core::grad(scaled_loss, parameter);
        const auto next = scaler.step(parameter, scaled_gradient, 0.1);

        ASSERT_TRUE(next.has_value());
        EXPECT_EQ(loss->dtype(), ormi::core::DType::float32);
        EXPECT_EQ(scaled_gradient->dtype(), ormi::core::DType::float32);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(scaled_gradient->data()), 24.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>((*next)->data()), 1.7f);
    }

    TEST(AutocastTest, rejectsUnsupportedTargetDtypes) {
        EXPECT_THROW(
            (void)ormi::core::Autocast(ormi::core::DType::float32),
            ormi::DTypeError
        );
        EXPECT_THROW(
            (void)ormi::core::Autocast(ormi::core::DType::int32),
            ormi::DTypeError
        );
    }
}
