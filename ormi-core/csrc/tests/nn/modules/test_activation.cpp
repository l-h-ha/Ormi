#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/nn/activation.h>
#include <ormi/nn/modules/activation.h>

#include "backends/cpu/pointwise.h"
#include "tests/nn/helpers.h"

namespace {
    TEST(NNActivationTest, directAndFusedKernelsAgree) {
        using namespace ormi::core;
        const auto input = nn_test::leaf({7});
        const std::array values{-1000., -2., -0.1, 0., 0.1, 2., 1000.};
        std::copy(values.begin(), values.end(), static_cast<double*>(input->data()));
        const std::array operations{
            ops::UnaryOp::abs,
            ops::UnaryOp::sign,
            ops::UnaryOp::positive,
            ops::UnaryOp::tanh,
            ops::UnaryOp::sigmoid,
            ops::UnaryOp::softplus,
            ops::UnaryOp::relu,
            ops::UnaryOp::erf,
            ops::UnaryOp::log1p,
            ops::UnaryOp::expm1
        };
        using Function = std::shared_ptr<Tensor> (*)(const std::shared_ptr<Tensor>&);
        const std::array<Function, 10> functions{
            ops::abs,
            ops::sign,
            ops::positive,
            ops::tanh,
            ops::sigmoid,
            ops::softplus,
            ops::relu,
            ops::erf,
            ops::log1p,
            ops::expm1
        };
        for (std::size_t operation = 0; operation < operations.size(); ++operation) {
            Tensor direct(Device::cpu(), {7}, DType::float64);
            ormi::backends::cpu::pointwise_unary(operations[operation], direct, *input);
            const auto fused = functions[operation](input);
            const auto* actual = static_cast<const double*>(fused->data());
            const auto* expected = static_cast<const double*>(direct.data());
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (std::isnan(expected[i])) {
                    EXPECT_TRUE(std::isnan(actual[i]));
                } else if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]);
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 1e-12);
                }
            }
        }
    }

    TEST(NNActivationTest, halfMaxGradientCountsTiesWithoutOverflow) {
        using namespace ormi::core;
        const auto input = std::make_shared<Tensor>(Device::cpu(), Shape{65536}, DType::float16);
        std::fill_n(static_cast<Float16*>(input->data()), input->numel(), Float16{1.f});
        const auto output = ops::max(input, {0});
        const auto gradient = grad(output, input);
        const auto* values = static_cast<const Float16*>(gradient->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            EXPECT_FLOAT_EQ(static_cast<float>(values[i]), 1.f / 65536.f);
        }
    }

    TEST(NNActivationTest, softmaxHandlesTiesAndDifferentiatesTwice) {
        const auto input = nn_test::leaf({2, 3}, 0.0);
        const auto probability = ormi::nn::Softmax{}(input);
        const auto gradient = ormi::core::grad(ormi::core::ops::sum(probability), input);
        const auto second = ormi::core::grad(ormi::core::ops::sum(gradient), input);
        EXPECT_FALSE(probability->realized());
        ormi::core::Plan plan({probability, gradient, second});
        plan.run();
        for (std::size_t i = 0; i < input->numel(); ++i) {
            EXPECT_NEAR(static_cast<const double*>(probability->data())[i], 1.0 / 3.0, 1e-12);
            EXPECT_NEAR(static_cast<const double*>(gradient->data())[i], 0.0, 1e-12);
            EXPECT_NEAR(static_cast<const double*>(second->data())[i], 0.0, 1e-12);
        }
    }
}
