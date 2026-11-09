#include <cmath>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/nn/losses/classification.h>
#include <ormi/nn/losses/regression.h>

#include "tests/nn/helpers.h"

namespace {
    TEST(NNLossTest, stableBinaryLogitsAndHuberBoundaries) {
        const auto input = nn_test::leaf({1}, -1000.);
        const auto target = nn_test::leaf({1}, 1.);
        const auto loss = ormi::nn::binary_cross_entropy_with_logits(input, target);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(loss->data()), 1000.);
        const auto gradient = ormi::core::grad(loss, input);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(gradient->data()), -1.);
        const auto boundary = nn_test::leaf({1}, 2.);
        const auto zero = nn_test::leaf({1}, 0.);
        const auto huber = ormi::nn::huber_loss(boundary, zero, 2.);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(huber->data()), 2.);
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(ormi::core::grad(huber, boundary)->data()), 2.);
    }

    TEST(NNLossTest, classAxisAndIgnoredTargets) {
        const auto logits = nn_test::leaf({2, 3}, 0.);
        const auto targets = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::int32
        );
        auto* labels = static_cast<std::int32_t*>(targets->data());
        labels[0] = 1;
        labels[1] = -100;
        const auto loss = ormi::nn::cross_entropy(logits, targets);
        const auto gradient = ormi::core::grad(loss, logits);
        EXPECT_NEAR(*static_cast<const double*>(loss->data()), std::log(3.), 1e-12);
        const auto* values = static_cast<const double*>(gradient->data());
        EXPECT_NEAR(values[0], 1. / 3., 1e-12);
        EXPECT_NEAR(values[1], -2. / 3., 1e-12);
        for (std::size_t i = 3; i < 6; ++i)
            EXPECT_DOUBLE_EQ(values[i], 0.);
    }
}
