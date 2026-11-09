#include <cmath>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/nn/modules/normalization.h>
#include <ormi/nn/normalization.h>

#include "tests/nn/helpers.h"

namespace {
    using namespace ormi::core;

    TEST(NNNormalizationTest, normalizesTrailingAxesAndOwnsLeafParameters) {
        const ormi::nn::LayerNorm layer({3, 4}, 0.01, true, true, DType::float64);
        EXPECT_EQ(layer.normalized_shape(), (Shape{3, 4}));
        EXPECT_DOUBLE_EQ(layer.eps(), 0.01);
        const auto parameters = layer.parameters();
        ASSERT_EQ(parameters.size(), 2u);
        EXPECT_EQ(parameters[0], layer.weight());
        EXPECT_EQ(parameters[1], layer.bias());
        for (const auto& parameter : parameters) {
            EXPECT_TRUE(parameter->realized());
            EXPECT_EQ(parameter->op(), nullptr);
        }

        const auto input = nn_test::leaf({2, 1, 3, 4});
        auto* values = static_cast<double*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<double>(i);
        }
        const auto output = layer(input);
        EXPECT_FALSE(output->realized());
        EXPECT_EQ(output->shape(), input->shape());
        const auto* result = static_cast<const double*>(output->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto centered = static_cast<double>(i % 12) - 5.5;
            EXPECT_NEAR(result[i], centered / std::sqrt(143.0 / 12.0 + 0.01), 1e-12);
        }
    }

    TEST(NNNormalizationTest, constantInputRetainsCorrectCurvature) {
        const auto input = nn_test::leaf({2}, 1.0);
        const auto direction = nn_test::leaf({2});
        static_cast<double*>(direction->data())[1] = -1.0;
        const auto output = ormi::nn::layer_norm(input, {2}, nullptr, nullptr, 0.25);
        const auto first = grad(ops::sum(ops::mul(output, direction)), input);
        const auto second = grad(ops::sum(ops::mul(first, direction)), input);
        const auto third = grad(ops::sum(ops::mul(second, direction)), input);
        Plan plan({output, first, second, third});
        plan.run();
        EXPECT_DOUBLE_EQ(static_cast<const double*>(output->data())[0], 0.0);
        EXPECT_DOUBLE_EQ(static_cast<const double*>(first->data())[0], 2.0);
        EXPECT_DOUBLE_EQ(static_cast<const double*>(second->data())[0], 0.0);
        EXPECT_DOUBLE_EQ(static_cast<const double*>(third->data())[0], -24.0);
    }

    TEST(NNNormalizationTest, optionalAffineAndMetadataErrors) {
        const ormi::nn::LayerNorm plain({3}, 1e-5, false);
        EXPECT_TRUE(plain.parameters().empty());
        EXPECT_EQ(plain.weight(), nullptr);
        EXPECT_EQ(plain.bias(), nullptr);
        const ormi::nn::LayerNorm no_bias({3}, 1e-5, true, false);
        EXPECT_EQ(no_bias.parameters().size(), 1u);
        EXPECT_EQ(no_bias.bias(), nullptr);
        const auto input = nn_test::leaf({2, 3});
        EXPECT_THROW(ormi::nn::LayerNorm(Shape{}), ormi::ShapeError);
        EXPECT_THROW(ormi::nn::LayerNorm(Shape{0}), ormi::ShapeError);
        EXPECT_THROW(ormi::nn::LayerNorm({3}, 0.0), ormi::OperationError);
        EXPECT_THROW(ormi::nn::LayerNorm({3}, std::numeric_limits<double>::infinity()), ormi::OperationError);
        EXPECT_THROW(ormi::nn::layer_norm(input, {2}), ormi::ShapeError);
        EXPECT_THROW(ormi::nn::layer_norm(nullptr, {3}), ormi::OperationError);
        EXPECT_THROW(ormi::nn::layer_norm(input, {3}, input), ormi::ShapeError);
    }
}
