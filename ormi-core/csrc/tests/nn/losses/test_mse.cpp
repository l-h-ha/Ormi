#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/nn/losses/mse.h>

#include "tests/nn/helpers.h"

namespace {
    TEST(NNLossTest, differentiatesMeanSquaredErrorTwiceWithoutRealizingTheRoot) {
        const auto prediction = nn_test::leaf({2}, 3.0);
        const auto target = nn_test::leaf({2}, 1.0);
        const auto loss = ormi::nn::mse_loss(prediction, target);
        const auto first = ormi::core::grad(loss, prediction);
        const auto second = ormi::core::grad(ormi::core::ops::sum(first), prediction);
        EXPECT_FALSE(loss->realized());
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(second->realized());
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(loss->data()), 4.0);
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(static_cast<const double*>(first->data())[i], 2.0);
            EXPECT_DOUBLE_EQ(static_cast<const double*>(second->data())[i], 1.0);
        }
    }

    TEST(NNLossTest, rejectsMissingMismatchedAndIntegerInputs) {
        const auto prediction = nn_test::leaf({2});
        EXPECT_THROW(ormi::nn::mse_loss(nullptr, prediction), ormi::OperationError);
        EXPECT_THROW(ormi::nn::mse_loss(prediction, nullptr), ormi::OperationError);
        EXPECT_THROW(ormi::nn::mse_loss(prediction, nn_test::leaf({1})), ormi::ShapeError);
        const auto integer = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::int32
        );
        EXPECT_THROW(ormi::nn::mse_loss(prediction, integer), ormi::DTypeError);
    }
}
