#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>

#include "nn/state.h"
#include "tests/nn/helpers.h"

namespace {
    void fail_alias(ormi::core::Tensor&) {
        throw std::runtime_error("state update failed");
    }

    constexpr ormi::core::Op failing_alias{
        .name = "failing_alias", .output = ormi::core::Output::alias,
        .parents = 1, .compute = fail_alias, .adjoint = nullptr, .scalar = {}
    };

    TEST(NNStateTest, failedAliasRemainsLazyAndLeavesParentStorageIntact) {
        const auto input = nn_test::leaf({2}, 3);
        const auto output = std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<failing_alias>(), std::vector{input}
        );
        for (int attempt = 0; attempt < 2; ++attempt) {
            EXPECT_THROW(output->realize(), std::runtime_error);
            EXPECT_FALSE(output->realized());
            EXPECT_DOUBLE_EQ(static_cast<const double*>(input->data())[0], 3);
        }
    }

    TEST(NNStateTest, aliasesStorageAndDoesNotDifferentiateTheWrite) {
        const auto input = nn_test::leaf({3}, 2);
        const auto destination = nn_test::leaf({2}, 0);
        const auto replacement = nn_test::leaf({2}, 7);
        const auto output = ormi::nn::detail::update(input, destination, replacement);
        const auto gradient = ormi::core::grad(ormi::core::ops::sum(output), input);
        ormi::core::Plan plan({gradient});
        plan.run();
        EXPECT_EQ(output->buffer(), input->buffer());
        EXPECT_EQ(output->data(), input->data());
        EXPECT_DOUBLE_EQ(static_cast<const double*>(destination->data())[0], 7);
        EXPECT_DOUBLE_EQ(static_cast<const double*>(gradient->data())[0], 1);
        EXPECT_EQ(plan.storage_buffers(), 1u); // Only the gradient's constant seed allocates.
    }

    TEST(NNStateTest, rejectsInvalidDestinationsAndDependencies) {
        const auto value = nn_test::leaf({2});
        EXPECT_THROW(ormi::nn::detail::update(value, nullptr, value), ormi::OperationError);
        EXPECT_THROW(ormi::nn::detail::update(value, value, nn_test::leaf({3})), ormi::ShapeError);
        EXPECT_THROW(ormi::nn::detail::depend(value, nullptr), ormi::OperationError);
    }
}
