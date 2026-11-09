#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/view.h>
#include <ormi/nn/modules/linear.h>

#include "tests/nn/helpers.h"

namespace {
    TEST(NNLinearTest, appliesToVectorsAndArbitraryBatchRanksLazily) {
        const auto weight = nn_test::leaf({3, 2});
        const auto bias = nn_test::leaf({2}, 0.25);
        const ormi::nn::Linear layer(weight, bias);
        for (const auto& shape : std::vector<ormi::core::Shape>{{3}, {4, 3}, {2, 4, 3}, {1, 2, 4, 3}}) {
            const auto input = nn_test::leaf(shape, 2.0);
            const auto output = layer(input);
            EXPECT_FALSE(output->realized());
            EXPECT_EQ(output->shape()[output->rank() - 1], 2);
            EXPECT_EQ(output->numel(), input->numel() / 3 * 2);
            const auto* data = static_cast<const double*>(output->data());
            for (std::size_t i = 0; i < output->numel(); ++i) {
                EXPECT_DOUBLE_EQ(data[i], 6.25);
            }
        }
        EXPECT_EQ(layer.weight(), weight);
        EXPECT_EQ(layer.bias(), bias);
        EXPECT_EQ(layer.in_features(), 3);
        EXPECT_EQ(layer.out_features(), 2);
    }

    TEST(NNLinearTest, acceptsStridedAndEmptyBatchInputsWithoutBias) {
        const ormi::nn::Linear layer(nn_test::leaf({3, 2}));
        EXPECT_EQ(layer.bias(), nullptr);
        EXPECT_EQ(layer.parameters().size(), 1u);
        const auto input = ormi::core::ops::transpose(nn_test::leaf({3, 4}, 2.0), {1, 0});
        const auto output = layer(input);
        EXPECT_EQ(output->shape(), (ormi::core::Shape{4, 2}));
        const auto* data = static_cast<const double*>(output->data());
        for (std::size_t i = 0; i < output->numel(); ++i) {
            EXPECT_DOUBLE_EQ(data[i], 6.0);
        }
        const auto empty = layer(nn_test::leaf({0, 3}));
        EXPECT_EQ(empty->shape(), (ormi::core::Shape{0, 2}));
        EXPECT_NO_THROW(empty->realize());
    }

    template <typename T>
    void check_initialization(ormi::core::DType dtype) {
        const auto device = ormi::core::Device::cpu();
        const ormi::nn::Linear first(3, 5, true, dtype, device, 17);
        const ormi::nn::Linear second(3, 5, true, dtype, device, 17);
        const auto parameters = first.parameters();
        const auto repeated = second.parameters();
        ASSERT_EQ(parameters.size(), 2u);
        EXPECT_EQ(parameters[0]->shape(), (ormi::core::Shape{3, 5}));
        EXPECT_EQ(parameters[1]->shape(), (ormi::core::Shape{5}));
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            EXPECT_TRUE(parameters[index]->realized());
            EXPECT_EQ(parameters[index]->op(), nullptr);
            EXPECT_EQ(parameters[index]->dtype(), dtype);
            const auto* data = static_cast<const T*>(parameters[index]->data());
            const auto* again = static_cast<const T*>(repeated[index]->data());
            const auto value = [](T element) {
                if constexpr (std::is_same_v<T, ormi::core::Float16>) {
                    return static_cast<double>(static_cast<float>(element));
                } else {
                    return static_cast<double>(element);
                }
            };
            for (std::size_t i = 0; i < parameters[index]->numel(); ++i) {
                EXPECT_EQ(value(data[i]), value(again[i]));
                EXPECT_LE(std::abs(value(data[i])), 1.0 / std::sqrt(3.0) + 1e-3);
            }
        }
    }

    TEST(NNLinearTest, initializesFloatingPointLeafParametersFromASeed) {
        check_initialization<float>(ormi::core::DType::float32);
        check_initialization<double>(ormi::core::DType::float64);
        check_initialization<ormi::core::Float16>(ormi::core::DType::float16);
    }

    TEST(NNLinearTest, rejectsInvalidFeaturesWeightsBiasesAndInputs) {
        EXPECT_THROW((ormi::nn::Linear(0, 2)), ormi::ShapeError);
        EXPECT_THROW((ormi::nn::Linear(2, -1)), ormi::ShapeError);
        EXPECT_THROW((ormi::nn::Linear(2, 3, true, ormi::core::DType::int32)), ormi::DTypeError);
        EXPECT_THROW((ormi::nn::Linear(nullptr)), ormi::ShapeError);
        EXPECT_THROW((ormi::nn::Linear(nn_test::leaf({2}))), ormi::ShapeError);
        const auto weight = nn_test::leaf({3, 2});
        EXPECT_THROW((ormi::nn::Linear(weight, nn_test::leaf({3}))), ormi::ShapeError);
        const auto bias = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        EXPECT_THROW((ormi::nn::Linear(weight, bias)), ormi::DTypeError);
        const ormi::nn::Linear layer(weight);
        EXPECT_THROW(layer(nullptr), ormi::OperationError);
        EXPECT_THROW(layer(nn_test::leaf({})), ormi::ShapeError);
        EXPECT_THROW(layer(nn_test::leaf({4})), ormi::ShapeError);
    }
}
