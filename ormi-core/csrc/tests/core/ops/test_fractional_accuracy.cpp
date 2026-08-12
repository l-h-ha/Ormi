#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/fractional.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    Tensor tensor(const std::vector<double>& values) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{
                static_cast<std::int64_t>(values.size())
            },
            ormi::core::DType::float64
        );
        auto* data = static_cast<double*>(result->data());
        std::copy(values.begin(), values.end(), data);
        return result;
    }

    Tensor scalar(double value) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float64
        );
        *static_cast<double*>(result->data()) = value;
        return result;
    }

    std::vector<double> samples(
        std::size_t count,
        double (*function)(double)
    ) {
        const auto spacing = 1.0 / static_cast<double>(count - 1);
        std::vector<double> result(count);

        for (std::size_t i = 0; i < count; ++i) {
            result[i] = function(spacing * static_cast<double>(i));
        }
        return result;
    }

    double quadratic(double time) {
        return time * time;
    }

    double quadratic_caputo_at_one(double alpha) {
        return std::tgamma(3.0) / std::tgamma(3.0 - alpha);
    }

    double final_error(std::size_t count, double alpha) {
        const auto spacing = 1.0 / static_cast<double>(count - 1);
        const auto input = tensor(samples(count, quadratic));
        const auto result =
            ormi::core::ops::caputo(input, alpha, spacing);
        const auto* output = static_cast<const double*>(result->data());
        return std::abs(
            output[count - 1] - quadratic_caputo_at_one(alpha)
        );
    }

    double cubic(double time) {
        return time * time * time;
    }

    double higher_order_final_error(std::size_t count, double alpha) {
        const auto spacing = 1.0 / static_cast<double>(count - 1);
        const auto input = tensor(samples(count, cubic));
        const auto result = ormi::core::ops::caputo(
            input,
            alpha,
            spacing,
            -1,
            scalar(0.0)
        );
        const auto expected = std::tgamma(4.0) / std::tgamma(4.0 - alpha);
        const auto* output = static_cast<const double*>(result->data());
        return std::abs(output[count - 1] - expected);
    }

    double dot(const double* lhs, const double* rhs, std::size_t count) {
        double result = 0.0;

        for (std::size_t i = 0; i < count; ++i) {
            result += lhs[i] * rhs[i];
        }
        return result;
    }

    TEST(FractionalAccuracyTest, annihilatesConstants) {
        constexpr std::size_t count = 129;
        const auto input = tensor(std::vector<double>(count, 3.25));

        for (const auto alpha : {0.2, 0.5, 0.8}) {
            const auto result = ormi::core::ops::caputo(
                input,
                alpha,
                1.0 / static_cast<double>(count - 1)
            );
            const auto* output =
                static_cast<const double*>(result->data());

            for (std::size_t i = 0; i < count; ++i) {
                EXPECT_DOUBLE_EQ(output[i], 0.0);
            }
        }
    }

    TEST(FractionalAccuracyTest, isCausalFromTheFirstSample) {
        constexpr std::size_t count = 33;
        auto changed_values = samples(count, quadratic);
        const auto original = tensor(changed_values);
        changed_values.back() += 1000.0;
        const auto changed = tensor(changed_values);
        const auto original_result =
            ormi::core::ops::caputo(
                original,
                0.4,
                1.0 / static_cast<double>(count - 1)
            );
        const auto changed_result =
            ormi::core::ops::caputo(
                changed,
                0.4,
                1.0 / static_cast<double>(count - 1)
            );
        const auto* original_output =
            static_cast<const double*>(original_result->data());
        const auto* changed_output =
            static_cast<const double*>(changed_result->data());

        for (std::size_t i = 0; i + 1 < count; ++i) {
            EXPECT_DOUBLE_EQ(original_output[i], changed_output[i]);
        }
        EXPECT_NE(original_output[count - 1], changed_output[count - 1]);
    }

    TEST(FractionalAccuracyTest, convergesAtTheL1Rate) {
        for (const auto alpha : {0.2, 0.5, 0.8}) {
            const auto coarse = final_error(65, alpha);
            const auto medium = final_error(129, alpha);
            const auto fine = final_error(257, alpha);

            EXPECT_GT(coarse, medium);
            EXPECT_GT(medium, fine);

            const auto observed = std::log2(medium / fine);
            EXPECT_NEAR(observed, 2.0 - alpha, 0.08);
        }
    }

    TEST(FractionalAccuracyTest, higherOrderSchemeConvergesOnCubicHistory) {
        for (const auto alpha : {1.2, 1.5, 1.8}) {
            const auto coarse = higher_order_final_error(65, alpha);
            const auto medium = higher_order_final_error(129, alpha);
            const auto fine = higher_order_final_error(257, alpha);

            EXPECT_GT(coarse, medium);
            EXPECT_GT(medium, fine);
            EXPECT_NEAR(std::log2(medium / fine), 1.0, 0.08);
        }
    }

    TEST(FractionalAccuracyTest, satisfiesTheAdjointIdentity) {
        constexpr std::size_t count = 31;
        std::vector<double> input_values(count);
        std::vector<double> seed_values(count);

        for (std::size_t i = 0; i < count; ++i) {
            const auto value = static_cast<double>(i);
            input_values[i] = std::sin(0.17 * value) + 0.01 * value * value;
            seed_values[i] = std::cos(0.11 * value) - 0.03 * value;
        }

        const auto input = tensor(input_values);
        const auto seed = tensor(seed_values);
        const auto output =
            ormi::core::ops::caputo(input, 0.63, 0.04);
        const auto transpose = ormi::core::grad(output, input, seed);
        const auto* output_data =
            static_cast<const double*>(output->data());
        const auto* transpose_data =
            static_cast<const double*>(transpose->data());
        const auto lhs = dot(output_data, seed_values.data(), count);
        const auto rhs = dot(input_values.data(), transpose_data, count);
        const auto scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});

        EXPECT_NEAR(lhs, rhs, 1e-12 * scale);
    }
}
