#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/optim/grad_scaler.h>
#include <ormi/core/optim/sgd.h>
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

    template <>
    ormi::core::DType dtype<ormi::core::Float16>() {
        return ormi::core::DType::float16;
    }

    class CUDAOptimizerTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            const cudaError_t status = cudaGetDeviceCount(&device_count);

            if (status != cudaSuccess) {
                GTEST_SKIP() << "CUDA runtime is unavailable: " << cudaGetErrorString(status);
            }

            if (device_count == 0) {
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
                ormi::core::Device::cuda(),
                bytes
            );
            ormi::core::memory::Buffer host(ormi::core::Device::cpu(), bytes);

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
        static std::vector<T> host_values(ormi::core::Tensor& tensor) {
            std::vector<T> values(tensor.numel());
            ormi::core::memory::Buffer host(
                ormi::core::Device::cpu(),
                values.size() * sizeof(T)
            );
            ormi::core::memory::copy(host, *tensor.buffer());

            if (host.bytes() != 0) {
                std::memcpy(values.data(), host.data(), host.bytes());
            }

            return values;
        }
    };

    TEST_F(CUDAOptimizerTest, realizesFunctionalFloat32Update) {
        const auto parameter = tensor<float>({3}, {1.0f, 2.0f, 3.0f});
        const auto gradient = tensor<float>({3}, {0.5f, 1.0f, -2.0f});
        auto next = ormi::core::optim::sgd(parameter, gradient, 0.1);

        EXPECT_FALSE(next->realized());
        EXPECT_EQ(
            host_values<float>(*next),
            (std::vector<float>{0.95f, 1.9f, 3.2f})
        );
        EXPECT_EQ(
            host_values<float>(*parameter),
            (std::vector<float>{1.0f, 2.0f, 3.0f})
        );
    }

    TEST_F(CUDAOptimizerTest, realizesFloat64AndFloat16Updates) {
        auto float64 = ormi::core::optim::sgd(
            tensor<double>({1}, {2.0}),
            tensor<double>({1}, {0.25}),
            0.5
        );
        auto float16 = ormi::core::optim::sgd(
            tensor<ormi::core::Float16>(
                {1},
                {ormi::core::Float16{2.0f}}
            ),
            tensor<ormi::core::Float16>(
                {1},
                {ormi::core::Float16{0.25f}}
            ),
            0.5
        );

        EXPECT_DOUBLE_EQ(host_values<double>(*float64)[0], 1.875);
        EXPECT_FLOAT_EQ(
            static_cast<float>(host_values<ormi::core::Float16>(*float16)[0]),
            1.875f
        );
    }

    TEST_F(CUDAOptimizerTest, readsTransposedParameterAndGradient) {
        const auto parameter = tensor<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
        const auto gradient = tensor<float>({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
        auto next = ormi::core::optim::sgd(
            ormi::core::ops::transpose(parameter, {1, 0}),
            ormi::core::ops::transpose(gradient, {1, 0}),
            0.01
        );

        EXPECT_EQ(
            host_values<float>(*next),
            (std::vector<float>{0.9f, 2.7f, 1.8f, 3.6f})
        );
    }

    TEST_F(CUDAOptimizerTest, fusesGradientUnscalingIntoUpdate) {
        const auto parameter = tensor<float>(ormi::core::Shape{}, {3.0f});
        const auto scaled_gradient = tensor<float>(ormi::core::Shape{}, {48.0f});
        auto next = ormi::core::optim::sgd(
            parameter,
            scaled_gradient,
            0.1,
            8.0
        );

        EXPECT_FLOAT_EQ(host_values<float>(*next)[0], 2.4f);
        EXPECT_EQ(next->parents()[1], scaled_gradient);
    }

    TEST_F(CUDAOptimizerTest, completesGradScalerFiniteAndOverflowPaths) {
        const std::vector<std::shared_ptr<ormi::core::Tensor>> parameters{
            tensor<float>(ormi::core::Shape{}, {3.0f}),
            tensor<float>(ormi::core::Shape{}, {5.0f})
        };
        const std::vector<std::shared_ptr<ormi::core::Tensor>> finite_gradients{
            tensor<float>(ormi::core::Shape{}, {48.0f}),
            tensor<float>(ormi::core::Shape{}, {16.0f})
        };
        ormi::core::optim::GradScaler scaler(8.0);
        const auto finite = scaler.step(parameters, finite_gradients, 0.1);

        ASSERT_TRUE(finite.has_value());
        EXPECT_FLOAT_EQ(host_values<float>(*(*finite)[0])[0], 2.4f);
        EXPECT_FLOAT_EQ(host_values<float>(*(*finite)[1])[0], 4.8f);

        const std::vector<std::shared_ptr<ormi::core::Tensor>> overflow_gradients{
            finite_gradients[0],
            tensor<float>(
                ormi::core::Shape{},
                {std::numeric_limits<float>::infinity()}
            )
        };
        EXPECT_FALSE(scaler.step(parameters, overflow_gradients, 0.1).has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 4.0);
    }

    TEST_F(CUDAOptimizerTest, updatesFloat32MasterFromFloat16Computation) {
        const auto parameter = tensor<float>(ormi::core::Shape{}, {3.0f});
        const auto half = ormi::core::ops::cast(
            parameter,
            ormi::core::DType::float16
        );
        const auto output = ormi::core::ops::mul(half, half);
        const auto gradient = ormi::core::grad(output, parameter);
        auto next = ormi::core::optim::sgd(parameter, gradient, 0.1);

        EXPECT_EQ(gradient->dtype(), ormi::core::DType::float32);
        EXPECT_FLOAT_EQ(host_values<float>(*next)[0], 2.4f);
        EXPECT_FLOAT_EQ(host_values<float>(*parameter)[0], 3.0f);
    }

    TEST_F(CUDAOptimizerTest, realizesEmptyOutputWithoutLaunching) {
        const auto parameter = ormi::core::ops::full(
            {0},
            1.0f,
            ormi::core::Device::cuda()
        );
        const auto gradient = ormi::core::ops::full(
            {0},
            2.0f,
            ormi::core::Device::cuda()
        );
        auto next = ormi::core::optim::sgd(parameter, gradient, 0.1);

        EXPECT_NO_THROW(next->realize());
        EXPECT_TRUE(next->realized());
        EXPECT_EQ(next->data(), nullptr);
    }

    TEST_F(CUDAOptimizerTest, limitsOnlyIndexedUpdateRank) {
        std::vector<std::int64_t> dims(65, 1);
        dims[0] = 2;
        const auto parameter = ormi::core::ops::expand(
            ormi::core::ops::full(
                ormi::core::Shape{},
                2.0f,
                ormi::core::Device::cuda()
            ),
            ormi::core::Shape{dims}
        );
        const auto gradient = ormi::core::ops::expand(
            ormi::core::ops::full(
                ormi::core::Shape{},
                1.0f,
                ormi::core::Device::cuda()
            ),
            ormi::core::Shape{dims}
        );
        auto indexed = ormi::core::optim::sgd(parameter, gradient, 0.1);
        auto contiguous = ormi::core::optim::sgd(
            ormi::core::ops::full(
                ormi::core::Shape{std::vector<std::int64_t>(65, 1)},
                2.0f,
                ormi::core::Device::cuda()
            ),
            ormi::core::ops::full(
                ormi::core::Shape{std::vector<std::int64_t>(65, 1)},
                1.0f,
                ormi::core::Device::cuda()
            ),
            0.1
        );

        EXPECT_FALSE(indexed->realized());
        EXPECT_THROW(indexed->realize(), ormi::ShapeError);
        EXPECT_FALSE(indexed->realized());
        EXPECT_FLOAT_EQ(host_values<float>(*contiguous)[0], 1.9f);
    }
}
