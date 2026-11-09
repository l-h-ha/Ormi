#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/plan.h>
#include <ormi/nn/losses/mse.h>
#include <ormi/nn/modules/activation.h>
#include <ormi/nn/modules/linear.h>
#include <ormi/nn/modules/sequential.h>

#include "helpers.h"

namespace {
    TEST(NNTrainingTest, differentiatesAComposedLayerThreeTimesAndThroughItsSecondDerivative) {
        const auto weight = nn_test::leaf({1, 1}, 3.0);
        const auto bias = nn_test::leaf({1}, 0.2);
        const auto input = nn_test::leaf({1}, 0.4);
        const ormi::nn::Sequential model(
            {std::make_shared<ormi::nn::Linear>(weight, bias), std::make_shared<ormi::nn::Sin>()}
        );
        const auto output = ormi::core::ops::sum(model(input));
        const auto first = ormi::core::grad(output, input);
        const auto second = ormi::core::grad(ormi::core::ops::sum(first), input);
        const auto third = ormi::core::grad(ormi::core::ops::sum(second), input);
        const auto mixed = ormi::core::grad(ormi::core::ops::sum(second), weight);
        EXPECT_FALSE(output->realized());
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(second->realized());
        EXPECT_FALSE(third->realized());
        EXPECT_FALSE(mixed->realized());
        EXPECT_NEAR(*static_cast<const double*>(first->data()), 3.0 * std::cos(1.4), 1e-12);
        EXPECT_NEAR(*static_cast<const double*>(second->data()), -9.0 * std::sin(1.4), 1e-12);
        EXPECT_NEAR(*static_cast<const double*>(third->data()), -27.0 * std::cos(1.4), 1e-12);
        EXPECT_NEAR(
            *static_cast<const double*>(mixed->data()),
            -6.0 * std::sin(1.4) - 3.6 * std::cos(1.4),
            1e-12
        );
    }

    TEST(NNTrainingTest, reusesAPlanAndStableParameterIdentitiesAcrossUpdates) {
        auto layer = std::make_shared<ormi::nn::Linear>(nn_test::leaf({1, 1}, 2.0));
        const auto parameters = layer->parameters();
        const auto loss = ormi::nn::mse_loss((*layer)(nn_test::leaf({4, 1})), nn_test::leaf({4, 1}, 0.0));
        const auto gradients = ormi::core::grad(loss, parameters);
        std::vector<ormi::core::Plan::Update> updates;
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            updates.push_back({parameters[i], ormi::core::optim::sgd(parameters[i], gradients[i], 0.1)});
        }
        ormi::core::Plan plan({loss}, updates);
        const auto steps = plan.steps();
        const auto buffers = plan.storage_buffers();
        const auto bytes = plan.storage_bytes();
        const auto* storage = parameters[0]->data();
        for (int i = 0; i < 8; ++i) {
            plan.run();
            EXPECT_NEAR(
                *static_cast<const double*>(parameters[0]->data()),
                2.0 * std::pow(0.8, i + 1),
                1e-12
            );
            EXPECT_EQ(layer->parameters()[0], parameters[0]);
            EXPECT_EQ(parameters[0]->data(), storage);
            EXPECT_EQ(parameters[0]->op(), nullptr);
        }
        EXPECT_EQ(plan.steps(), steps);
        EXPECT_EQ(plan.storage_buffers(), buffers);
        EXPECT_EQ(plan.storage_bytes(), bytes);
    }

    TEST(NNTrainingTest, graphOutlivesTheLayerThatBuiltIt) {
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::nn::Linear layer(nn_test::leaf({2, 1}, 3.0), nn_test::leaf({1}, 0.5));
            output = layer(nn_test::leaf({2}, 2.0));
        }
        EXPECT_DOUBLE_EQ(*static_cast<const double*>(output->data()), 12.5);
    }
}
