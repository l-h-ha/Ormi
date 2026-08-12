#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/fractional.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    template <typename T>
    ormi::core::DType dtype();

    template <>
    ormi::core::DType dtype<double>() {
        return ormi::core::DType::float64;
    }

    template <>
    ormi::core::DType dtype<float>() {
        return ormi::core::DType::float32;
    }

    class CUDAFractionalTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            const auto status = cudaGetDeviceCount(&devices);

            if (status != cudaSuccess) {
                GTEST_SKIP()
                    << "CUDA runtime is unavailable: "
                    << cudaGetErrorString(status);
            }
            if (devices == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
        }

        template <typename T>
        static std::shared_ptr<ormi::core::Tensor> tensor(
            ormi::core::Shape shape,
            const std::vector<T>& values
        ) {
            const auto bytes = values.size() * sizeof(T);
            auto device_buffer = std::make_shared<ormi::core::memory::Buffer>(
                ormi::core::Device::from("cuda"),
                bytes
            );
            ormi::core::memory::Buffer host(
                ormi::core::Device::cpu(),
                bytes
            );

            if (bytes != 0) {
                std::memcpy(host.data(), values.data(), bytes);
                ormi::core::memory::copy(*device_buffer, host);
            }

            return std::make_shared<ormi::core::Tensor>(
                std::move(device_buffer),
                std::move(shape),
                dtype<T>()
            );
        }

        template <typename T>
        static std::vector<T> host_values(ormi::core::Tensor& value) {
            std::vector<T> result(value.numel());
            ormi::core::memory::Buffer host(
                ormi::core::Device::cpu(),
                result.size() * sizeof(T)
            );
            ormi::core::memory::copy(host, *value.buffer());

            if (host.bytes() != 0) {
                std::memcpy(result.data(), host.data(), host.bytes());
            }
            return result;
        }

        template <typename T>
        static std::shared_ptr<ormi::core::Tensor> cpu_tensor(
            ormi::core::Shape shape,
            const std::vector<T>& values
        ) {
            auto result = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(),
                std::move(shape),
                dtype<T>()
            );

            if (!values.empty()) {
                std::memcpy(
                    result->data(),
                    values.data(),
                    values.size() * sizeof(T)
                );
            }
            return result;
        }

        template <typename T>
        static void expect_cpu_parity(double tolerance) {
            constexpr std::size_t lines = 2;
            constexpr std::size_t history = 257;
            constexpr double spacing = 0.01;
            std::vector<T> values(lines * history);
            std::vector<T> seeds(lines * history);

            for (std::size_t line = 0; line < lines; ++line) {
                for (std::size_t n = 0; n < history; ++n) {
                    const auto time = spacing * static_cast<double>(n);
                    values[line * history + n] = static_cast<T>(
                        std::sin(static_cast<double>(line + 1) * time) +
                        0.1 * std::cos(0.3 * time) +
                        0.002 * time * time
                    );
                    seeds[line * history + n] = static_cast<T>(
                        std::cos(0.2 * static_cast<double>(line + 1) * time) -
                        0.001 * static_cast<double>(n)
                    );
                }
            }

            const auto cpu_input = ormi::core::ops::transpose(
                cpu_tensor<T>({2, 257}, values),
                {1, 0}
            );
            const auto cuda_input = ormi::core::ops::transpose(
                tensor<T>({2, 257}, values),
                {1, 0}
            );
            const auto cpu_result =
                ormi::core::ops::caputo(cpu_input, 0.73, spacing, 0);
            const auto cuda_result =
                ormi::core::ops::caputo(cuda_input, 0.73, spacing, 0);
            const auto expected = host_values<T>(*cpu_result);
            const auto actual = host_values<T>(*cuda_result);

            ASSERT_EQ(actual.size(), expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i) {
                const auto scale = std::max(
                    1.0,
                    std::abs(static_cast<double>(expected[i]))
                );
                EXPECT_NEAR(
                    static_cast<double>(actual[i]),
                    static_cast<double>(expected[i]),
                    tolerance * scale
                );
            }

            const auto cpu_seed = ormi::core::ops::transpose(
                cpu_tensor<T>({2, 257}, seeds),
                {1, 0}
            );
            const auto cuda_seed = ormi::core::ops::transpose(
                tensor<T>({2, 257}, seeds),
                {1, 0}
            );
            const auto cpu_gradient =
                ormi::core::grad(cpu_result, cpu_input, cpu_seed);
            const auto cuda_gradient =
                ormi::core::grad(cuda_result, cuda_input, cuda_seed);
            const auto expected_gradient = host_values<T>(*cpu_gradient);
            const auto actual_gradient = host_values<T>(*cuda_gradient);

            ASSERT_EQ(actual_gradient.size(), expected_gradient.size());
            for (std::size_t i = 0; i < expected_gradient.size(); ++i) {
                const auto scale = std::max(
                    1.0,
                    std::abs(static_cast<double>(expected_gradient[i]))
                );
                EXPECT_NEAR(
                    static_cast<double>(actual_gradient[i]),
                    static_cast<double>(expected_gradient[i]),
                    tolerance * scale
                );
            }
        }

        template <typename T>
        static void expect_higher_order_cpu_parity(double tolerance) {
            constexpr std::size_t lines = 2;
            constexpr std::size_t history = 129;
            constexpr double spacing = 0.01;
            std::vector<T> values(lines * history);
            std::vector<T> seeds(lines * history);
            std::vector<T> directions(lines * history);
            const std::vector<T> initial{
                static_cast<T>(1.0),
                static_cast<T>(2.0)
            };

            for (std::size_t line = 0; line < lines; ++line) {
                for (std::size_t n = 0; n < history; ++n) {
                    const auto time = spacing * static_cast<double>(n);
                    values[line * history + n] = static_cast<T>(
                        static_cast<double>(line + 1) * time +
                        0.5 * time * time +
                        std::sin(0.2 * time)
                    );
                    seeds[line * history + n] = static_cast<T>(
                        std::cos(0.13 * static_cast<double>(n)) - 0.01 * time
                    );
                    directions[line * history + n] = static_cast<T>(
                        std::sin(0.07 * static_cast<double>(n)) + 0.2
                    );
                }
            }

            const auto compare = [tolerance](
                const char* phase,
                const std::vector<T>& expected,
                const std::vector<T>& actual
            ) {
                SCOPED_TRACE(phase);
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < expected.size(); ++i) {
                    const auto scale = std::max(
                        1.0,
                        std::abs(static_cast<double>(expected[i]))
                    );
                    EXPECT_NEAR(
                        static_cast<double>(actual[i]),
                        static_cast<double>(expected[i]),
                        tolerance * scale
                    );
                }
            };

            const auto cpu_input = ormi::core::ops::transpose(
                cpu_tensor<T>({2, 129}, values),
                {1, 0}
            );
            const auto cuda_input = ormi::core::ops::transpose(
                tensor<T>({2, 129}, values),
                {1, 0}
            );
            const auto cpu_initial = cpu_tensor<T>({2}, initial);
            const auto cuda_initial = tensor<T>({2}, initial);
            const auto cpu_result = ormi::core::ops::caputo(
                cpu_input,
                1.47,
                spacing,
                0,
                cpu_initial
            );
            const auto cuda_result = ormi::core::ops::caputo(
                cuda_input,
                1.47,
                spacing,
                0,
                cuda_initial
            );
            compare(
                "forward",
                host_values<T>(*cpu_result),
                host_values<T>(*cuda_result)
            );

            const auto cpu_seed = ormi::core::ops::transpose(
                cpu_tensor<T>({2, 129}, seeds),
                {1, 0}
            );
            const auto cuda_seed = ormi::core::ops::transpose(
                tensor<T>({2, 129}, seeds),
                {1, 0}
            );
            const auto cpu_gradients = ormi::core::grad(
                cpu_result,
                std::vector<std::shared_ptr<ormi::core::Tensor>>{
                    cpu_input,
                    cpu_initial
                },
                cpu_seed
            );
            const auto cuda_gradients = ormi::core::grad(
                cuda_result,
                std::vector<std::shared_ptr<ormi::core::Tensor>>{
                    cuda_input,
                    cuda_initial
                },
                cuda_seed
            );
            compare(
                "history adjoint",
                host_values<T>(*cpu_gradients[0]),
                host_values<T>(*cuda_gradients[0])
            );
            compare(
                "initial derivative adjoint",
                host_values<T>(*cpu_gradients[1]),
                host_values<T>(*cuda_gradients[1])
            );

            const auto cpu_direction = ormi::core::ops::transpose(
                cpu_tensor<T>({2, 129}, directions),
                {1, 0}
            );
            const auto cuda_direction = ormi::core::ops::transpose(
                tensor<T>({2, 129}, directions),
                {1, 0}
            );
            const auto cpu_history_second = ormi::core::grad(
                cpu_gradients[0],
                cpu_seed,
                cpu_direction
            );
            const auto cuda_history_second = ormi::core::grad(
                cuda_gradients[0],
                cuda_seed,
                cuda_direction
            );
            compare(
                "history transpose adjoint",
                host_values<T>(*cpu_history_second),
                host_values<T>(*cuda_history_second)
            );

            const std::vector<T> initial_direction{
                static_cast<T>(1),
                static_cast<T>(1)
            };
            const auto cpu_initial_second = ormi::core::grad(
                cpu_gradients[1],
                cpu_seed,
                cpu_tensor<T>({2}, initial_direction)
            );
            const auto cuda_initial_second = ormi::core::grad(
                cuda_gradients[1],
                cuda_seed,
                tensor<T>({2}, initial_direction)
            );
            compare(
                "initial transpose adjoint",
                host_values<T>(*cpu_initial_second),
                host_values<T>(*cuda_initial_second)
            );
        }

        static std::vector<double> matrix(
            std::size_t count,
            double alpha,
            double spacing
        ) {
            std::vector<double> result(count * count, 0.0);
            const auto scale =
                std::pow(spacing, -alpha) / std::tgamma(2.0 - alpha);

            for (std::size_t n = 1; n < count; ++n) {
                for (std::size_t k = 0; k < n; ++k) {
                    const auto weight = scale * (
                        std::pow(
                            static_cast<double>(k + 1),
                            1.0 - alpha
                        ) -
                        std::pow(static_cast<double>(k), 1.0 - alpha)
                    );
                    const auto column = n - k;
                    result[n * count + column] += weight;
                    result[n * count + column - 1] -= weight;
                }
            }
            return result;
        }

        static std::vector<double> transpose_product(
            const std::vector<double>& matrix,
            const std::vector<double>& values
        ) {
            const auto count = values.size();
            std::vector<double> result(count, 0.0);

            for (std::size_t row = 0; row < count; ++row) {
                for (std::size_t column = 0; column < count; ++column) {
                    result[row] +=
                        matrix[column * count + row] * values[column];
                }
            }
            return result;
        }

        static std::vector<double> product(
            const std::vector<double>& matrix,
            const std::vector<double>& values
        ) {
            const auto count = values.size();
            std::vector<double> result(count, 0.0);

            for (std::size_t row = 0; row < count; ++row) {
                for (std::size_t column = 0; column < count; ++column) {
                    result[row] +=
                        matrix[row * count + column] * values[column];
                }
            }
            return result;
        }
    };

    TEST_F(CUDAFractionalTest, realizesFloat64AlongAStridedAxis) {
        constexpr double alpha = 0.5;
        constexpr double spacing = 0.25;
        const auto input = tensor<double>(
            {2, 4},
            {
                1.0, 1.25, 1.5, 1.75,
                2.0, 2.5, 3.0, 3.5
            }
        );
        const auto transposed = ormi::core::ops::transpose(input, {1, 0});
        const auto result =
            ormi::core::ops::caputo(transposed, alpha, spacing, 0);

        EXPECT_FALSE(result->realized());
        const auto actual = host_values<double>(*result);
        const auto gamma = std::tgamma(2.0 - alpha);

        for (std::size_t n = 0; n < 4; ++n) {
            for (std::size_t line = 0; line < 2; ++line) {
                const auto slope = static_cast<double>(line + 1);
                const auto time = spacing * static_cast<double>(n);
                const auto expected = n == 0
                    ? 0.0
                    : slope * std::pow(time, 1.0 - alpha) / gamma;
                EXPECT_NEAR(actual[n * 2 + line], expected, 1e-12);
            }
        }
    }

    TEST_F(CUDAFractionalTest, realizesTheExactDiscreteAdjoint) {
        constexpr double alpha = 0.4;
        constexpr double spacing = 0.2;
        const std::vector<double> seed_values{
            0.5, -1.0, 2.0, 0.25, 3.0
        };
        const auto input = tensor<double>(
            {5},
            {1.0, 1.5, 2.0, 4.0, 3.0}
        );
        const auto seed = tensor<double>({5}, seed_values);
        const auto output =
            ormi::core::ops::caputo(input, alpha, spacing);
        const auto gradient = ormi::core::grad(output, input, seed);

        EXPECT_FALSE(gradient->realized());
        const auto actual = host_values<double>(*gradient);
        const auto expected = transpose_product(
            matrix(seed_values.size(), alpha, spacing),
            seed_values
        );

        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-12);
        }
    }

    TEST_F(CUDAFractionalTest, supportsFloat32AndDegenerateHistories) {
        const auto input = tensor<float>(
            {4},
            {2.0f, 2.5f, 3.0f, 3.5f}
        );
        const auto result = ormi::core::ops::caputo(input, 0.5, 0.5);
        const auto actual = host_values<float>(*result);
        const auto gamma = std::tgamma(1.5);

        for (std::size_t n = 0; n < actual.size(); ++n) {
            const auto expected = n == 0
                ? 0.0
                : std::sqrt(0.5 * static_cast<double>(n)) / gamma;
            EXPECT_NEAR(actual[n], expected, 1e-6);
        }

        const auto single = tensor<float>({2, 1}, {4.0f, -2.0f});
        const auto single_result =
            ormi::core::ops::caputo(single, 0.25, 0.1);
        EXPECT_EQ(
            host_values<float>(*single_result),
            (std::vector<float>{0.0f, 0.0f})
        );

        const auto empty = tensor<float>({2, 0, 3}, {});
        const auto empty_result =
            ormi::core::ops::caputo(empty, 0.25, 0.1, 1);
        empty_result->realize();
        EXPECT_TRUE(empty_result->realized());
        EXPECT_EQ(empty_result->numel(), 0);

        const auto higher_single_initial =
            tensor<float>({2}, {1.0f, -1.0f});
        const auto higher_single = ormi::core::ops::caputo(
            single,
            1.5,
            0.1,
            -1,
            higher_single_initial
        );
        EXPECT_EQ(
            host_values<float>(*higher_single),
            (std::vector<float>{0.0f, 0.0f})
        );

        const auto higher_empty_initial = tensor<float>(
            {2, 3},
            std::vector<float>(6, 0.5f)
        );
        const auto higher_empty = ormi::core::ops::caputo(
            empty,
            1.5,
            0.1,
            1,
            higher_empty_initial
        );
        const auto empty_seed = tensor<float>({2, 0, 3}, {});
        const auto empty_initial_gradient = ormi::core::grad(
            higher_empty,
            higher_empty_initial,
            empty_seed
        );
        EXPECT_EQ(
            host_values<float>(*empty_initial_gradient),
            std::vector<float>(6, 0.0f)
        );
    }

    TEST_F(CUDAFractionalTest, differentiatesThroughTheAdjoint) {
        constexpr double alpha = 0.6;
        constexpr double spacing = 0.125;
        const std::vector<double> direction{
            -1.0, 0.5, 2.0, -0.25, 1.5
        };
        const auto input = tensor<double>(
            {5},
            {1.0, 2.0, 4.0, 8.0, 16.0}
        );
        const auto seed = tensor<double>(
            {5},
            {0.5, 1.0, -2.0, 3.0, 0.25}
        );
        const auto weights = tensor<double>({5}, direction);
        const auto output =
            ormi::core::ops::caputo(input, alpha, spacing);
        const auto transpose = ormi::core::grad(output, input, seed);
        const auto objective = ormi::core::ops::sum(
            ormi::core::ops::mul(transpose, weights)
        );
        const auto second = ormi::core::grad(objective, seed);

        EXPECT_FALSE(second->realized());
        const auto actual = host_values<double>(*second);
        const auto expected = product(
            matrix(direction.size(), alpha, spacing),
            direction
        );

        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-12);
        }
    }

    TEST_F(CUDAFractionalTest, matchesCPUForLongStridedHistories) {
        expect_cpu_parity<double>(2e-12);
        expect_cpu_parity<float>(2e-5);
    }

    TEST_F(CUDAFractionalTest, matchesCPUAboveOrderOne) {
        expect_higher_order_cpu_parity<double>(2e-12);
        expect_higher_order_cpu_parity<float>(1e-4);
    }

    TEST_F(CUDAFractionalTest, recordsWarmedPlans) {
        const auto input = tensor<float>(
            {4},
            {1.0f, 1.25f, 1.5f, 1.75f}
        );
        const auto result = ormi::core::ops::caputo(input, 0.5, 0.25);
        std::optional<ormi::core::Plan::Replay> replay;

        {
            ormi::core::Plan plan({result});
            EXPECT_EQ(plan.storage_buffers(), 2);
            EXPECT_EQ(
                plan.storage_bytes(),
                4 * sizeof(float) + 3 * sizeof(double)
            );
            plan.run();
            replay.emplace(plan.record());
        }

        replay->run();
        const auto actual = host_values<float>(*result);
        EXPECT_NEAR(actual.back(), std::sqrt(0.75) / std::tgamma(1.5), 1e-6);

        constexpr std::size_t count = 64;
        constexpr double spacing = 0.01;
        constexpr double alpha = 1.5;
        std::vector<double> history(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto time = static_cast<double>(i) * spacing;
            history[i] = time * time;
        }

        const auto higher_input = tensor<double>({count}, history);
        const auto initial_derivative = tensor<double>(
            ormi::core::Shape{},
            {0.0}
        );
        const auto higher_result = ormi::core::ops::caputo(
            higher_input,
            alpha,
            spacing,
            -1,
            initial_derivative
        );
        std::optional<ormi::core::Plan::Replay> higher_replay;

        {
            ormi::core::Plan plan({higher_result});
            EXPECT_EQ(plan.storage_buffers(), 3);
            EXPECT_EQ(
                plan.storage_bytes(),
                count * sizeof(double) +
                    (count - 1) * sizeof(double) +
                    count * sizeof(double)
            );
            plan.run();
            higher_replay.emplace(plan.record());
        }

        higher_replay->run();
        const auto higher_actual = host_values<double>(*higher_result);
        const auto final_time = static_cast<double>(count - 1) * spacing;
        const auto expected =
            2.0 * std::pow(final_time, 2.0 - alpha) /
            std::tgamma(3.0 - alpha);
        EXPECT_NEAR(higher_actual.back(), expected, 1e-12);
    }
}
