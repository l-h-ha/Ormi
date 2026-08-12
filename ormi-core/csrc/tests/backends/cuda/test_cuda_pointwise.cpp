#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    template <typename T>
    ormi::core::DType dtype();

    template <>
    ormi::core::DType dtype<float>() {
        return ormi::core::DType::float32;
    }

    template <>
    ormi::core::DType dtype<double>() {
        return ormi::core::DType::float64;
    }

    template <>
    ormi::core::DType dtype<ormi::core::Float16>() {
        return ormi::core::DType::float16;
    }

    template <>
    ormi::core::DType dtype<std::int32_t>() {
        return ormi::core::DType::int32;
    }

    class CUDAPointwiseTest : public ::testing::Test {
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
            const std::vector<T>& values,
            std::vector<std::int64_t> strides = {}
        ) {
            const auto bytes = values.size() * sizeof(T);
            auto device_buffer = std::make_shared<ormi::core::memory::Buffer>(
                ormi::core::Device::from("cuda"),
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
                dtype<T>(),
                std::move(strides)
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

    TEST_F(CUDAPointwiseTest, realizesEveryBinaryOperation) {
        const auto lhs = tensor<float>({2}, {8.0f, 6.0f});
        const auto rhs = tensor<float>({2}, {2.0f, 3.0f});

        auto add = ormi::core::ops::add(lhs, rhs);
        auto sub = ormi::core::ops::sub(lhs, rhs);
        auto mul = ormi::core::ops::mul(lhs, rhs);
        auto div = ormi::core::ops::div(lhs, rhs);

        EXPECT_FALSE(add->realized());
        EXPECT_EQ(host_values<float>(*add), (std::vector<float>{10.0f, 9.0f}));
        EXPECT_EQ(host_values<float>(*sub), (std::vector<float>{6.0f, 3.0f}));
        EXPECT_EQ(host_values<float>(*mul), (std::vector<float>{16.0f, 18.0f}));
        EXPECT_EQ(host_values<float>(*div), (std::vector<float>{4.0f, 2.0f}));
    }

    TEST_F(CUDAPointwiseTest, realizesMultiaxisBroadcast) {
        const auto lhs = tensor<float>(
            {2, 1, 3},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
        );
        const auto rhs = tensor<float>({1, 4, 1}, {10.0f, 20.0f, 30.0f, 40.0f});
        auto result = ormi::core::ops::add(lhs, rhs);

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 4, 3}));
        EXPECT_EQ(
            host_values<float>(*result),
            (std::vector<float>{
                11.0f, 12.0f, 13.0f,
                21.0f, 22.0f, 23.0f,
                31.0f, 32.0f, 33.0f,
                41.0f, 42.0f, 43.0f,
                14.0f, 15.0f, 16.0f,
                24.0f, 25.0f, 26.0f,
                34.0f, 35.0f, 36.0f,
                44.0f, 45.0f, 46.0f
            })
        );
    }

    TEST_F(CUDAPointwiseTest, realizesStridedInputs) {
        const auto storage = tensor<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
        const auto transposed = ormi::core::ops::transpose(storage, {1, 0});
        const auto rhs = tensor<float>({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});

        auto binary = ormi::core::ops::add(transposed, rhs);
        auto unary = ormi::core::ops::neg(transposed);

        EXPECT_EQ(
            host_values<float>(*binary),
            (std::vector<float>{11.0f, 23.0f, 32.0f, 44.0f})
        );
        EXPECT_EQ(
            host_values<float>(*unary),
            (std::vector<float>{-1.0f, -3.0f, -2.0f, -4.0f})
        );
    }

    TEST_F(CUDAPointwiseTest, realizesEveryNumericDType) {
        auto float64 = ormi::core::ops::mul(
            tensor<double>({2}, {1.5, 2.0}),
            tensor<double>({2}, {2.0, 4.0})
        );
        EXPECT_EQ(host_values<double>(*float64), (std::vector<double>{3.0, 8.0}));

        auto float16 = ormi::core::ops::add(
            tensor<ormi::core::Float16>(
                {2},
                {ormi::core::Float16{1.5f}, ormi::core::Float16{2.0f}}
            ),
            tensor<ormi::core::Float16>(
                {2},
                {ormi::core::Float16{0.5f}, ormi::core::Float16{3.0f}}
            )
        );
        const auto float16_values = host_values<ormi::core::Float16>(*float16);
        EXPECT_FLOAT_EQ(static_cast<float>(float16_values[0]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(float16_values[1]), 5.0f);

        auto int32 = ormi::core::ops::sub(
            tensor<std::int32_t>({2}, {7, 4}),
            tensor<std::int32_t>({2}, {2, 6})
        );
        EXPECT_EQ(host_values<std::int32_t>(*int32), (std::vector<std::int32_t>{5, -2}));
    }

    TEST_F(CUDAPointwiseTest, realizesEveryUnaryOperation) {
        const auto input = tensor<float>({3}, {0.5f, 1.0f, 2.0f});
        auto neg = ormi::core::ops::neg(input);
        auto exp = ormi::core::ops::exp(input);
        auto log = ormi::core::ops::log(input);
        auto sin = ormi::core::ops::sin(input);
        auto cos = ormi::core::ops::cos(input);

        const auto neg_values = host_values<float>(*neg);
        const auto exp_values = host_values<float>(*exp);
        const auto log_values = host_values<float>(*log);
        const auto sin_values = host_values<float>(*sin);
        const auto cos_values = host_values<float>(*cos);

        for (std::size_t i = 0; i < 3; ++i) {
            const auto value = std::vector<float>{0.5f, 1.0f, 2.0f}[i];
            EXPECT_FLOAT_EQ(neg_values[i], -value);
            EXPECT_NEAR(exp_values[i], std::exp(value), 1e-6f);
            EXPECT_NEAR(log_values[i], std::log(value), 1e-6f);
            EXPECT_NEAR(sin_values[i], std::sin(value), 1e-6f);
            EXPECT_NEAR(cos_values[i], std::cos(value), 1e-6f);
        }
    }

    TEST_F(CUDAPointwiseTest, realizesFloat16TranscendentalAndInt32Negation) {
        auto float16 = ormi::core::ops::exp(
            tensor<ormi::core::Float16>({1}, {ormi::core::Float16{1.0f}})
        );
        const auto float16_value = host_values<ormi::core::Float16>(*float16)[0];
        EXPECT_NEAR(static_cast<float>(float16_value), std::exp(1.0f), 0.002f);

        auto int32 = ormi::core::ops::neg(tensor<std::int32_t>({3}, {-2, 0, 7}));
        EXPECT_EQ(
            host_values<std::int32_t>(*int32),
            (std::vector<std::int32_t>{2, 0, -7})
        );
    }

    TEST_F(CUDAPointwiseTest, realizesLazySecondDerivative) {
        const auto input = tensor<float>(ormi::core::Shape{}, {3.0f});
        const auto square = ormi::core::ops::mul(input, input);
        const auto cube = ormi::core::ops::mul(square, input);
        auto second = ormi::core::diff(cube, input, 2);

        EXPECT_FALSE(second->realized());
        EXPECT_FLOAT_EQ(host_values<float>(*second)[0], 18.0f);
    }

    TEST_F(CUDAPointwiseTest, realizesEmptyOutputWithoutLaunching) {
        const auto lhs = ormi::core::ops::full({0, 3}, 2.0f, ormi::core::Device::from("cuda"));
        const auto rhs = ormi::core::ops::full({1, 3}, 4.0f, ormi::core::Device::from("cuda"));
        auto result = ormi::core::ops::add(lhs, rhs);

        EXPECT_NO_THROW(result->realize());
        EXPECT_TRUE(result->realized());
        EXPECT_EQ(result->data(), nullptr);
    }

    TEST_F(CUDAPointwiseTest, rejectsOversizedIndexedRankLazily) {
        const ormi::core::Shape shape(std::vector<std::int64_t>(65, 1));
        const auto lhs = ormi::core::ops::full(shape, 2.0f, ormi::core::Device::from("cuda"));
        const auto rhs = ormi::core::ops::full(
            ormi::core::Shape{},
            3.0f,
            ormi::core::Device::from("cuda")
        );
        auto result = ormi::core::ops::add(lhs, rhs);

        EXPECT_FALSE(result->realized());
        EXPECT_THROW(result->realize(), ormi::ShapeError);
        EXPECT_FALSE(result->realized());
    }
}
