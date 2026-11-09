#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>

#include "backends/cpu/pointwise.h"

namespace {
    using namespace ormi::core;

    TEST(RsqrtTest, directAndFusedFloat64PreserveRangeAndSpecialValues) {
        const std::array values{
            1e-300,
            0.25,
            2.0,
            1e300,
            0.0,
            -0.0,
            -1.0,
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN()
        };
        const auto input = std::make_shared<Tensor>(Device::cpu(), Shape{9}, DType::float64);
        std::copy(values.begin(), values.end(), static_cast<double*>(input->data()));
        Tensor direct(Device::cpu(), {9}, DType::float64);
        ormi::backends::cpu::pointwise_unary(ops::UnaryOp::rsqrt, direct, *input);
        const auto fused = ops::rsqrt(input);
        EXPECT_FALSE(fused->realized());
        const auto* optimized = static_cast<const double*>(fused->data());
        const auto* ordinary = static_cast<const double*>(direct.data());

        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto expected = 1.0 / std::sqrt(values[i]);
            if (std::isnan(expected)) {
                EXPECT_TRUE(std::isnan(ordinary[i]));
                EXPECT_TRUE(std::isnan(optimized[i]));
                continue;
            }
            EXPECT_DOUBLE_EQ(ordinary[i], expected);
            EXPECT_DOUBLE_EQ(optimized[i], expected);
            EXPECT_EQ(std::signbit(optimized[i]), std::signbit(expected));
        }
    }

    TEST(RsqrtTest, directHalfHandlesStrides) {
        const auto input = std::make_shared<Tensor>(Device::cpu(), Shape{2, 3}, DType::float16);
        const std::array values{0.25f, 1.f, 4.f, 9.f, 16.f, 25.f};
        auto* stored = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < values.size(); ++i) {
            stored[i] = Float16{values[i]};
        }
        const auto transposed = ops::transpose(input, {1, 0});
        Tensor output(Device::cpu(), {3, 2}, DType::float16);
        ormi::backends::cpu::pointwise_unary(ops::UnaryOp::rsqrt, output, *transposed);
        const auto* result = static_cast<const Float16*>(output.data());
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                EXPECT_NEAR(static_cast<float>(result[i * 2 + j]), 1.f / std::sqrt(values[j * 3 + i]), 5e-4f);
            }
        }
    }

    TEST(RsqrtTest, differentiatesThreeTimesLazily) {
        const auto input = std::make_shared<Tensor>(Device::cpu(), Shape{3}, DType::float64);
        const std::array values{0.25, 1.0, 4.0};
        std::copy(values.begin(), values.end(), static_cast<double*>(input->data()));
        std::vector<std::shared_ptr<Tensor>> roots{ops::rsqrt(input)};
        for (int order = 0; order < 3; ++order) {
            roots.push_back(grad(ops::sum(roots.back()), input));
            EXPECT_FALSE(roots.back()->realized());
        }
        Plan plan(roots);
        plan.run();
        const std::array factors{1.0, -0.5, 0.75, -1.875};
        for (std::size_t order = 0; order < roots.size(); ++order) {
            const auto* result = static_cast<const double*>(roots[order]->data());
            for (std::size_t i = 0; i < values.size(); ++i) {
                EXPECT_DOUBLE_EQ(result[i], factors[order] * std::pow(values[i], -0.5 - order));
            }
        }
    }

    TEST(RsqrtTest, rejectsMissingAndNonfloatingInputs) {
        EXPECT_THROW(ops::rsqrt(nullptr), ormi::OperationError);
        const auto input = std::make_shared<Tensor>(Device::cpu(), Shape{1}, DType::int32);
        EXPECT_THROW(ops::rsqrt(input), ormi::DTypeError);
        Tensor output(Device::cpu(), {1}, DType::int32);
        EXPECT_THROW(
            ormi::backends::cpu::pointwise_unary(ops::UnaryOp::rsqrt, output, *input),
            ormi::DTypeError
        );
    }
}
