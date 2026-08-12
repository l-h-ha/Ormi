#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/attrs.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/fractional.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
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

        for (std::size_t i = 0; i < values.size(); ++i) {
            data[i] = values[i];
        }
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

    std::vector<double> l1_matrix(
        std::size_t count,
        double alpha,
        double spacing
    ) {
        std::vector<double> matrix(count * count, 0.0);
        const auto scale =
            std::pow(spacing, -alpha) / std::tgamma(2.0 - alpha);

        for (std::size_t n = 1; n < count; ++n) {
            for (std::size_t k = 0; k < n; ++k) {
                const auto weight = scale * (
                    std::pow(static_cast<double>(k + 1), 1.0 - alpha) -
                    std::pow(static_cast<double>(k), 1.0 - alpha)
                );
                const auto column = n - k;
                matrix[n * count + column] += weight;
                matrix[n * count + column - 1] -= weight;
            }
        }
        return matrix;
    }

    std::vector<double> l2_matrix(
        std::size_t count,
        double alpha,
        double spacing
    ) {
        std::vector<double> matrix(count * count, 0.0);
        const auto scale =
            std::pow(spacing, -alpha) / std::tgamma(3.0 - alpha);

        for (std::size_t n = 1; n < count; ++n) {
            for (std::size_t k = 0; k < n; ++k) {
                const auto weight = scale * (
                    std::pow(static_cast<double>(k + 1), 2.0 - alpha) -
                    std::pow(static_cast<double>(k), 2.0 - alpha)
                );
                const auto difference = n - 1 - k;

                if (difference == 0) {
                    matrix[n * count] -= 2.0 * weight;
                    matrix[n * count + 1] += 2.0 * weight;
                } else {
                    matrix[n * count + difference - 1] += weight;
                    matrix[n * count + difference] -= 2.0 * weight;
                    matrix[n * count + difference + 1] += weight;
                }
            }
        }
        return matrix;
    }

    std::vector<double> l2_initial_column(
        std::size_t count,
        double alpha,
        double spacing
    ) {
        std::vector<double> result(count, 0.0);
        const auto scale =
            std::pow(spacing, -alpha) / std::tgamma(3.0 - alpha);

        for (std::size_t n = 1; n < count; ++n) {
            const auto weight = scale * (
                std::pow(static_cast<double>(n), 2.0 - alpha) -
                std::pow(static_cast<double>(n - 1), 2.0 - alpha)
            );
            result[n] = -2.0 * spacing * weight;
        }
        return result;
    }

    std::vector<double> multiply(
        const std::vector<double>& matrix,
        const std::vector<double>& values,
        bool transpose
    ) {
        const auto count = values.size();
        std::vector<double> result(count, 0.0);

        for (std::size_t row = 0; row < count; ++row) {
            for (std::size_t column = 0; column < count; ++column) {
                const auto index = transpose
                    ? column * count + row
                    : row * count + column;
                result[row] += matrix[index] * values[column];
            }
        }
        return result;
    }

    TEST(AutogradFractionalTest, usesExactDiscreteTranspose) {
        constexpr double alpha = 0.4;
        constexpr double spacing = 0.2;
        const std::vector<double> seed_values{
            0.5,
            -1.0,
            2.0,
            0.25,
            3.0
        };
        const auto input = tensor({1.0, 1.5, 2.0, 4.0, 3.0});
        const auto seed = tensor(seed_values);
        const auto output =
            ormi::core::ops::caputo(input, alpha, spacing);
        const auto gradient = ormi::core::grad(output, input, seed);

        EXPECT_FALSE(gradient->realized());
        EXPECT_EQ(
            std::get<ormi::core::Caputo>(output->attrs()).coefficients,
            std::get<ormi::core::Caputo>(gradient->attrs()).coefficients
        );

        const auto matrix =
            l1_matrix(seed_values.size(), alpha, spacing);
        const auto expected =
            multiply(matrix, seed_values, true);
        const auto* actual =
            static_cast<const double*>(gradient->data());

        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-12);
        }
    }

    TEST(AutogradFractionalTest, differentiatesThroughTheTranspose) {
        constexpr double alpha = 0.6;
        constexpr double spacing = 0.125;
        const std::vector<double> direction{
            -1.0,
            0.5,
            2.0,
            -0.25,
            1.5
        };
        const auto input = tensor({1.0, 2.0, 4.0, 8.0, 16.0});
        const auto seed = tensor({0.5, 1.0, -2.0, 3.0, 0.25});
        const auto weights = tensor(direction);
        const auto output =
            ormi::core::ops::caputo(input, alpha, spacing);
        const auto transpose = ormi::core::grad(output, input, seed);
        const auto objective = ormi::core::ops::sum(
            ormi::core::ops::mul(transpose, weights)
        );
        const auto second = ormi::core::grad(objective, seed);

        const auto matrix =
            l1_matrix(direction.size(), alpha, spacing);
        const auto expected = multiply(matrix, direction, false);
        const auto* actual =
            static_cast<const double*>(second->data());

        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-12);
        }
    }

    TEST(AutogradFractionalTest, differentiatesAQuadraticResidual) {
        constexpr double alpha = 0.5;
        constexpr double spacing = 0.25;
        const std::vector<double> input_values{
            1.0,
            1.5,
            3.0,
            2.0,
            5.0
        };
        const auto input = tensor(input_values);
        const auto derivative =
            ormi::core::ops::caputo(input, alpha, spacing);
        const auto loss = ormi::core::ops::sum(
            ormi::core::ops::mul(derivative, derivative)
        );
        const auto gradient = ormi::core::grad(loss, input);

        const auto matrix =
            l1_matrix(input_values.size(), alpha, spacing);
        auto expected = multiply(matrix, input_values, false);
        for (auto& value : expected) {
            value *= 2.0;
        }
        expected = multiply(matrix, expected, true);

        const auto* actual =
            static_cast<const double*>(gradient->data());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-11);
        }
    }

    TEST(AutogradFractionalTest, transposesBothHigherOrderInputsExactly) {
        constexpr double alpha = 1.4;
        constexpr double spacing = 0.2;
        const std::vector<double> seed_values{
            0.5,
            -1.0,
            2.0,
            0.25,
            3.0
        };
        const auto input = tensor({1.0, 1.5, 2.0, 4.0, 3.0});
        const auto initial_derivative = scalar(0.75);
        const auto seed = tensor(seed_values);
        const auto output = ormi::core::ops::caputo(
            input,
            alpha,
            spacing,
            -1,
            initial_derivative
        );
        const auto gradients = ormi::core::grad(
            output,
            std::vector<Tensor>{input, initial_derivative},
            seed
        );

        EXPECT_FALSE(gradients[0]->realized());
        EXPECT_FALSE(gradients[1]->realized());

        const auto matrix = l2_matrix(seed_values.size(), alpha, spacing);
        const auto expected_history =
            multiply(matrix, seed_values, true);
        const auto initial_column =
            l2_initial_column(seed_values.size(), alpha, spacing);
        double expected_initial = 0.0;
        for (std::size_t i = 0; i < seed_values.size(); ++i) {
            expected_initial += initial_column[i] * seed_values[i];
        }

        const auto* history =
            static_cast<const double*>(gradients[0]->data());
        for (std::size_t i = 0; i < expected_history.size(); ++i) {
            EXPECT_NEAR(history[i], expected_history[i], 1e-12);
        }
        EXPECT_NEAR(
            *static_cast<const double*>(gradients[1]->data()),
            expected_initial,
            1e-12
        );
    }

    TEST(AutogradFractionalTest, differentiatesThroughHigherOrderTransposes) {
        constexpr double alpha = 1.6;
        constexpr double spacing = 0.125;
        const std::vector<double> direction{
            -1.0,
            0.5,
            2.0,
            -0.25,
            1.5
        };
        const auto input = tensor({1.0, 2.0, 4.0, 8.0, 16.0});
        const auto initial_derivative = scalar(0.25);
        const auto seed = tensor({0.5, 1.0, -2.0, 3.0, 0.25});
        const auto weights = tensor(direction);
        const auto output = ormi::core::ops::caputo(
            input,
            alpha,
            spacing,
            -1,
            initial_derivative
        );

        const auto history_transpose = ormi::core::grad(output, input, seed);
        const auto history_objective = ormi::core::ops::sum(
            ormi::core::ops::mul(history_transpose, weights)
        );
        const auto history_second =
            ormi::core::grad(history_objective, seed);
        const auto matrix = l2_matrix(direction.size(), alpha, spacing);
        const auto expected_history = multiply(matrix, direction, false);
        const auto* actual_history =
            static_cast<const double*>(history_second->data());

        for (std::size_t i = 0; i < expected_history.size(); ++i) {
            EXPECT_NEAR(actual_history[i], expected_history[i], 1e-12);
        }

        const auto initial_transpose =
            ormi::core::grad(output, initial_derivative, seed);
        const auto initial_second =
            ormi::core::grad(initial_transpose, seed);
        const auto expected_initial =
            l2_initial_column(direction.size(), alpha, spacing);
        const auto* actual_initial =
            static_cast<const double*>(initial_second->data());

        for (std::size_t i = 0; i < expected_initial.size(); ++i) {
            EXPECT_NEAR(actual_initial[i], expected_initial[i], 1e-12);
        }
    }
}
