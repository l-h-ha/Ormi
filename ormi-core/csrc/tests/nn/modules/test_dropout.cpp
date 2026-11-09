#include <algorithm>
#include <cstddef>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/nn/dropout.h>
#include <ormi/nn/modules/dropout.h>

#include "tests/nn/helpers.h"

namespace {
    TEST(NNDropoutTest, seededLayersMatchAndPreparedExecutionDrawsFreshMasks) {
        const auto input = nn_test::leaf({1024});
        ormi::nn::Dropout first(0.5, 93);
        ormi::nn::Dropout second(0.5, 93);
        const auto a = first(input);
        const auto b = second(input);
        ormi::core::Plan plan({a, b});
        plan.run();
        const auto* av = static_cast<const double*>(a->data());
        const auto* bv = static_cast<const double*>(b->data());
        EXPECT_TRUE(std::equal(av, av + input->numel(), bv));
        const std::vector<double> saved(av, av + input->numel());
        plan.run();
        EXPECT_TRUE(std::equal(av, av + input->numel(), bv));
        EXPECT_FALSE(std::equal(av, av + input->numel(), saved.begin()));
    }

    TEST(NNDropoutTest, gradientsReuseTheForwardMask) {
        const auto input = nn_test::leaf({1024});
        const auto output = ormi::nn::dropout(input, 0.5, true, 47);
        const auto gradient = ormi::core::grad(ormi::core::ops::sum(output), input);
        ormi::core::Plan plan({output, gradient});
        plan.run();
        const auto* values = static_cast<const double*>(output->data());
        const auto* gradients = static_cast<const double*>(gradient->data());
        EXPECT_TRUE(std::equal(values, values + input->numel(), gradients));
    }

    TEST(NNDropoutTest, evaluationAndZeroProbabilityReturnInput) {
        const auto input = nn_test::leaf({4});
        ormi::nn::Dropout layer;
        layer.eval();
        EXPECT_EQ(layer(input), input);
        EXPECT_EQ(ormi::nn::dropout(input, 0), input);
        EXPECT_THROW(ormi::nn::Dropout(-0.1), ormi::OperationError);
        EXPECT_THROW(ormi::nn::Dropout(1.1), ormi::OperationError);
    }
}
