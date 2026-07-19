#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    TEST(AutogradReductionTest, differentiatesSumOfSquares) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float32
        );
        auto* input_data = static_cast<float*>(input->data());

        for (std::size_t i = 0; i < 4; ++i) {
            input_data[i] = static_cast<float>(i + 1);
        }

        const auto squared = ormi::core::ops::mul(input, input);
        const auto output = ormi::core::ops::sum(squared);
        const auto gradient = ormi::core::grad(output, input);
        const auto* data = static_cast<const float*>(gradient->data());

        EXPECT_FLOAT_EQ(data[0], 2.0f);
        EXPECT_FLOAT_EQ(data[1], 4.0f);
        EXPECT_FLOAT_EQ(data[2], 6.0f);
        EXPECT_FLOAT_EQ(data[3], 8.0f);
    }

    TEST(AutogradReductionTest, differentiatesFloat64SumOfSquares) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float64
        );
        auto* input_data = static_cast<double*>(input->data());
        input_data[0] = 1.5;
        input_data[1] = 2.5;

        const auto output = ormi::core::ops::sum(ormi::core::ops::mul(input, input));
        const auto gradient = ormi::core::grad(output, input);
        const auto* data = static_cast<const double*>(gradient->data());

        EXPECT_DOUBLE_EQ(data[0], 3.0);
        EXPECT_DOUBLE_EQ(data[1], 5.0);
    }
}
