#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/nn/modules/batch_norm.h>

#include "tests/nn/helpers.h"

namespace {
    TEST(NNBatchNormTest, ownsSeparateParametersAndRunningStatistics) {
        ormi::nn::BatchNorm layer(3);
        EXPECT_EQ(layer.parameters(), (std::vector{layer.weight(), layer.bias()}));
        EXPECT_EQ(layer.buffers(), (std::vector{layer.running_mean(), layer.running_var()}));
        EXPECT_TRUE(layer.training());
        EXPECT_EQ(layer.axis(), -1);
        EXPECT_EQ(layer.num_features(), 3);
        EXPECT_EQ(layer.eps(), 1e-5);
        EXPECT_EQ(layer.momentum(), 0.1);
        for (const auto& buffer : layer.buffers()) {
            EXPECT_TRUE(buffer->realized());
            EXPECT_EQ(buffer->op(), nullptr);
        }
    }

    TEST(NNBatchNormTest, updatesOnlyWhenItsLazyGraphExecutes) {
        using namespace ormi::core;
        ormi::nn::BatchNorm layer(2, 1e-5, 0.5, false, true, -1, DType::float64);
        auto input = nn_test::leaf({2, 2});
        auto* data = static_cast<double*>(input->data());
        data[0] = 1;
        data[1] = 2;
        data[2] = 3;
        data[3] = 6;
        auto output = layer(input);
        auto* mean = static_cast<const double*>(layer.running_mean()->data());
        EXPECT_EQ(mean[0], 0);
        EXPECT_FALSE(output->realized());
        Plan plan({output});
        plan.run();
        EXPECT_DOUBLE_EQ(mean[0], 1);
        EXPECT_DOUBLE_EQ(mean[1], 2);
        plan.run();
        EXPECT_DOUBLE_EQ(mean[0], 1.5);
        EXPECT_DOUBLE_EQ(mean[1], 3);
        layer.eval();
        auto inference = layer(input);
        inference->realize();
        EXPECT_DOUBLE_EQ(mean[0], 1.5);
    }

    TEST(NNBatchNormTest, releasesCompletedGraphsRatherThanGrowingAHistory) {
        using namespace ormi::core;
        ormi::nn::BatchNorm layer(2, 1e-5, 0.1, false, true, -1, DType::float64);
        auto input = nn_test::leaf({2, 2});
        auto first = layer(input);
        std::weak_ptr<Tensor> lifetime = first;
        first->realize();
        auto second = layer(input);
        first.reset();
        EXPECT_TRUE(lifetime.expired());
        second->realize();
    }

    TEST(NNBatchNormTest, rejectsInvalidAxesAndInsufficientSamples) {
        ormi::nn::BatchNorm layer(3, 1e-5, 0.1, true, true, -1, ormi::core::DType::float64);
        EXPECT_THROW(layer(nn_test::leaf({1, 3})), ormi::ShapeError);
        EXPECT_THROW(layer(nn_test::leaf({2, 4})), ormi::ShapeError);
        EXPECT_THROW(ormi::nn::BatchNorm(0), ormi::ShapeError);
        EXPECT_THROW(ormi::nn::BatchNorm(3, 1e-5, -1), ormi::OperationError);
    }

    TEST(NNBatchNormTest, rejectsStateDTypeMismatchesWithoutExecutingTheInput) {
        using namespace ormi::core;
        for (const bool affine : {false, true}) {
            for (const auto dtype : {DType::float16, DType::float32}) {
                ormi::nn::BatchNorm layer(2, 1e-5, 0.1, affine, true, -1, DType::float64);
                const auto input = ops::cast(nn_test::leaf({2, 2}), dtype);

                EXPECT_FALSE(input->realized());
                EXPECT_THROW(layer(input), ormi::DTypeError);
                EXPECT_FALSE(input->realized());
                const auto* mean = static_cast<const double*>(layer.running_mean()->data());
                const auto* variance = static_cast<const double*>(layer.running_var()->data());
                for (std::size_t i = 0; i < 2; ++i) {
                    EXPECT_DOUBLE_EQ(mean[i], 0);
                    EXPECT_DOUBLE_EQ(variance[i], 1);
                }
            }
        }
    }
}
