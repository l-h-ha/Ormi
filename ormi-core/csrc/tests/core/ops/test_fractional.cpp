#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include <ormi/core/attrs.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/fractional.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> values(
        ormi::core::Shape shape
    ) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float64
        );
    }

    TEST(FractionalTest, constructsOneLazyCaputoNode) {
        const auto input = values({2, 5});
        const auto result =
            ormi::core::ops::caputo(input, 0.5, 0.25);

        ASSERT_FALSE(result->realized());
        EXPECT_EQ(result->shape(), input->shape());
        EXPECT_EQ(result->dtype(), input->dtype());
        EXPECT_TRUE(ormi::core::ops::is_caputo(*result));

        const auto& attrs =
            std::get<ormi::core::Caputo>(result->attrs());
        EXPECT_DOUBLE_EQ(attrs.alpha, 0.5);
        EXPECT_DOUBLE_EQ(attrs.spacing, 0.25);
        EXPECT_EQ(attrs.axis, 1);
        EXPECT_EQ(attrs.coefficients->forward.size(), 4);
        EXPECT_EQ(attrs.coefficients->adjoint.size(), 4);
    }

    TEST(FractionalTest, differentiatesLinearHistoriesAlongAnyAxis) {
        constexpr double alpha = 0.5;
        constexpr double spacing = 0.25;
        const auto input = values({2, 5});
        auto* input_data = static_cast<double*>(input->data());

        for (std::size_t row = 0; row < 2; ++row) {
            const auto slope = static_cast<double>(row + 1);
            for (std::size_t n = 0; n < 5; ++n) {
                input_data[row * 5 + n] =
                    3.0 + slope * spacing * static_cast<double>(n);
            }
        }

        const auto result =
            ormi::core::ops::caputo(input, alpha, spacing, -1);
        const auto* output = static_cast<const double*>(result->data());
        const auto gamma = std::tgamma(2.0 - alpha);

        for (std::size_t row = 0; row < 2; ++row) {
            const auto slope = static_cast<double>(row + 1);
            for (std::size_t n = 0; n < 5; ++n) {
                const auto time = spacing * static_cast<double>(n);
                const auto expected = n == 0
                    ? 0.0
                    : slope * std::pow(time, 1.0 - alpha) / gamma;
                EXPECT_NEAR(output[row * 5 + n], expected, 1e-12);
            }
        }

        const auto transposed =
            ormi::core::ops::transpose(input, {1, 0});
        const auto transposed_result =
            ormi::core::ops::caputo(transposed, alpha, spacing, 0);
        const auto* transposed_output =
            static_cast<const double*>(transposed_result->data());

        for (std::size_t n = 0; n < 5; ++n) {
            for (std::size_t row = 0; row < 2; ++row) {
                EXPECT_NEAR(
                    transposed_output[n * 2 + row],
                    output[row * 5 + n],
                    1e-12
                );
            }
        }
    }

    TEST(FractionalTest, constructsOneLazyHigherOrderNode) {
        const auto input = values({2, 5, 3});
        const auto initial_derivative = values({2, 3});
        const auto result = ormi::core::ops::caputo(
            input,
            1.5,
            0.25,
            1,
            initial_derivative
        );

        EXPECT_FALSE(result->realized());
        EXPECT_EQ(result->parents().size(), 2);
        EXPECT_EQ(result->parents()[0], input);
        EXPECT_EQ(result->parents()[1], initial_derivative);
        EXPECT_TRUE(ormi::core::ops::is_caputo(*result));

        const auto& attrs = std::get<ormi::core::Caputo>(result->attrs());
        EXPECT_DOUBLE_EQ(attrs.alpha, 1.5);
        EXPECT_EQ(attrs.axis, 1);
        EXPECT_EQ(attrs.count, 5);
        EXPECT_EQ(attrs.coefficients->forward.size(), 4);
        EXPECT_EQ(attrs.coefficients->adjoint.size(), 4);
    }

    TEST(FractionalTest, differentiatesQuadraticHistoriesAboveOrderOne) {
        constexpr std::size_t count = 9;
        constexpr double spacing = 0.125;
        const auto input = values({2, static_cast<std::int64_t>(count)});
        const auto initial_derivative = values({2});
        auto* input_data = static_cast<double*>(input->data());
        auto* initial_data = static_cast<double*>(initial_derivative->data());

        for (std::size_t row = 0; row < 2; ++row) {
            const auto linear = static_cast<double>(row + 1);
            const auto quadratic = static_cast<double>(row + 2);
            initial_data[row] = linear;

            for (std::size_t n = 0; n < count; ++n) {
                const auto time = spacing * static_cast<double>(n);
                input_data[row * count + n] =
                    3.0 + linear * time + quadratic * time * time;
            }
        }

        for (const auto alpha : {1.2, 1.5, 1.8}) {
            const auto result = ormi::core::ops::caputo(
                input,
                alpha,
                spacing,
                -1,
                initial_derivative
            );
            const auto* output = static_cast<const double*>(result->data());

            for (std::size_t row = 0; row < 2; ++row) {
                const auto quadratic = static_cast<double>(row + 2);
                for (std::size_t n = 0; n < count; ++n) {
                    const auto time = spacing * static_cast<double>(n);
                    const auto expected = n == 0
                        ? 0.0
                        : 2.0 * quadratic *
                            std::pow(time, 2.0 - alpha) /
                            std::tgamma(3.0 - alpha);
                    EXPECT_NEAR(output[row * count + n], expected, 1e-11);
                }
            }
        }
    }

    TEST(FractionalTest, handlesEmptyAndSingleSampleHistories) {
        const auto empty = values({2, 0, 3});
        const auto empty_result =
            ormi::core::ops::caputo(empty, 0.25, 0.1, 1);

        empty_result->realize();
        EXPECT_TRUE(empty_result->realized());
        EXPECT_EQ(empty_result->numel(), 0);

        const auto single = values({2, 1});
        auto* single_data = static_cast<double*>(single->data());
        single_data[0] = 4.0;
        single_data[1] = -2.0;

        const auto single_result =
            ormi::core::ops::caputo(single, 0.25, 0.1);
        const auto* output =
            static_cast<const double*>(single_result->data());
        EXPECT_DOUBLE_EQ(output[0], 0.0);
        EXPECT_DOUBLE_EQ(output[1], 0.0);
    }

    TEST(FractionalTest, supportsFloat32Histories) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float32
        );
        auto* input_data = static_cast<float*>(input->data());
        input_data[0] = 2.0f;
        input_data[1] = 2.5f;
        input_data[2] = 3.0f;
        input_data[3] = 3.5f;

        const auto result =
            ormi::core::ops::caputo(input, 0.5, 0.5);
        const auto* output =
            static_cast<const float*>(result->data());
        const auto gamma = std::tgamma(1.5);

        for (std::size_t n = 0; n < 4; ++n) {
            const auto expected = n == 0
                ? 0.0
                : std::sqrt(0.5 * static_cast<double>(n)) / gamma;
            EXPECT_NEAR(output[n], expected, 1e-6);
        }
    }

    TEST(FractionalTest, rejectsUndefinedParametersAndDtypes) {
        const auto input = values({4});
        const auto scalar = values({});
        const auto integer = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::int32
        );

        EXPECT_THROW(
            ormi::core::ops::caputo(nullptr, 0.5, 0.1),
            ormi::OperationError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(scalar, 0.5, 0.1),
            ormi::ShapeError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(input, 0.0, 0.1),
            ormi::OperationError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(input, 1.0, 0.1),
            ormi::OperationError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(input, 2.0, 0.1),
            ormi::OperationError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(input, 1.5, 0.1),
            ormi::OperationError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(input, 0.5, 0.1, -1, values({})),
            ormi::OperationError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(input, 1.5, 0.1, -1, values({1})),
            ormi::ShapeError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(
                input,
                std::numeric_limits<double>::quiet_NaN(),
                0.1
            ),
            ormi::OperationError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(input, 0.5, 0.0),
            ormi::OperationError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(input, 0.5, 0.1, 1),
            ormi::ShapeError
        );
        EXPECT_THROW(
            ormi::core::ops::caputo(integer, 0.5, 0.1),
            ormi::DTypeError
        );
    }
}
