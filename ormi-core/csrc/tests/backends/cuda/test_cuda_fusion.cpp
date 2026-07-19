#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/codegen.h"
#include "core/runtime/fusion/expr.h"

namespace {
    TEST(CUDAFusionCodegenTest, emitsDeterministicFloat32Kernel) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto first = ormi::backends::cuda::fusion::emit(*expression);
        const auto second = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_EQ(first, second);
        EXPECT_NE(first.find("extern \"C\" __global__ void ormi_kernel"), std::string::npos);
        EXPECT_NE(first.find("const float v1 = v0 * v0;"), std::string::npos);
        EXPECT_NE(first.find("const float v2 = v1 + v0;"), std::string::npos);
        EXPECT_NE(first.find("const float v3 = sinf(v2);"), std::string::npos);
        EXPECT_NE(first.find("output[i] = v3;"), std::string::npos);
    }

    TEST(CUDAFusionCodegenTest, rejectsMalformedExpression) {
        const ormi::core::runtime::fusion::Expr malformed{
            ormi::core::DType::float32,
            {},
            {},
            0,
            0
        };

        EXPECT_THROW(
            (void)ormi::backends::cuda::fusion::emit(malformed),
            ormi::OperationError
        );
    }

    class CUDAFusionTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            const auto status = cudaGetDeviceCount(&device_count);

            if (status != cudaSuccess) {
                GTEST_SKIP() << "CUDA runtime is unavailable: " << cudaGetErrorString(status);
            }
            if (device_count == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
        }

        template <typename T>
        static std::shared_ptr<ormi::core::Tensor> tensor(
            ormi::core::DType dtype,
            const std::vector<T>& values
        ) {
            const auto bytes = values.size() * sizeof(T);
            auto device_buffer = std::make_shared<ormi::core::memory::Buffer>(
                ormi::core::Device::cuda(),
                bytes
            );
            ormi::core::memory::Buffer host(ormi::core::Device::cpu(), bytes);
            std::memcpy(host.data(), values.data(), bytes);
            ormi::core::memory::copy(*device_buffer, host);
            return std::make_shared<ormi::core::Tensor>(
                std::move(device_buffer),
                ormi::core::Shape{static_cast<std::int64_t>(values.size())},
                dtype
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
            std::memcpy(values.data(), host.data(), host.bytes());
            return values;
        }
    };

    TEST_F(CUDAFusionTest, executesFloat32ChainWithoutRealizingIntermediates) {
        const auto input = tensor<float>(
            ormi::core::DType::float32,
            {-0.5f, 0.0f, 0.5f, 1.0f}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto sum = ormi::core::ops::add(square, input);
        const auto exponential = ormi::core::ops::exp(sum);
        const auto output = ormi::core::ops::sin(exponential);
        const auto values = host_values<float>(*output);

        for (std::size_t i = 0; i < values.size(); ++i) {
            const float x = static_cast<float>(i) * 0.5f - 0.5f;
            EXPECT_NEAR(values[i], std::sin(std::exp(x * x + x)), 1.0e-6f);
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(sum->realized());
        EXPECT_FALSE(exponential->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST_F(CUDAFusionTest, executesFloat64ChainAndReusesItsKernel) {
        for (int repetition = 0; repetition < 2; ++repetition) {
            const auto input = tensor<double>(ormi::core::DType::float64, {0.25, 0.5});
            const auto square = ormi::core::ops::mul(input, input);
            const auto output = ormi::core::ops::cos(ormi::core::ops::exp(square));
            const auto values = host_values<double>(*output);

            EXPECT_NEAR(values[0], std::cos(std::exp(0.0625)), 1.0e-12);
            EXPECT_NEAR(values[1], std::cos(std::exp(0.25)), 1.0e-12);
            EXPECT_FALSE(square->realized());
        }
    }

    TEST_F(CUDAFusionTest, fallsBackForFloat16) {
        using ormi::core::Float16;
        const auto input = tensor<Float16>(
            ormi::core::DType::float16,
            {Float16{1.0f}, Float16{2.0f}}
        );
        const auto doubled = ormi::core::ops::add(input, input);
        const auto output = ormi::core::ops::mul(doubled, input);
        const auto values = host_values<Float16>(*output);

        EXPECT_FLOAT_EQ(static_cast<float>(values[0]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(values[1]), 8.0f);
        EXPECT_TRUE(doubled->realized());
    }

}
