#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/pointwise.h"
#include "core/runtime/dispatch/memory.h"
#include "core/runtime/dispatch/pointwise.h"

namespace {
    TEST(CUDANNTest, directRsqrtSupportsAllFloatingDtypes) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "No CUDA device is available.";
        }

        using namespace ormi::core;
        const auto device = Device::from("cuda:0");
        for (const auto dtype : {DType::float16, DType::float32, DType::float64}) {
            const auto input = ops::cast(ops::full({8}, 2.0f, device), dtype);
            const auto output = std::make_shared<Tensor>(device, Shape{8}, dtype);
            runtime::pointwise::unary(ops::UnaryOp::rsqrt, *output, *input);
            const auto converted = ops::cast(output, DType::float64);
            std::array<double, 8> values{};
            runtime::memory::copy(Device::cpu(), values.data(), device, converted->data(), sizeof(values));

            double tolerance = 3e-15;
            if (dtype == DType::float16) {
                tolerance = 3e-4;
            } else if (dtype == DType::float32) {
                tolerance = 1e-7;
            }
            for (const auto value : values) {
                EXPECT_NEAR(value, 1.0 / std::sqrt(2.0), tolerance);
            }
        }
    }

    TEST(CUDANNTest, halfMaxGradientCountsTiesWithoutOverflow) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "No CUDA device is available.";
        }
        using namespace ormi::core;
        const auto device = Device::from("cuda:0");
        const auto input = ops::full({65536}, Float16{1.f}, device);
        const auto output = ops::max(input, {0});
        const auto gradient = grad(output, input);
        std::vector<Float16> values(input->numel());
        runtime::memory::copy(
            Device::cpu(),
            values.data(),
            device,
            gradient->data(),
            values.size() * sizeof(Float16)
        );
        for (const auto value : values) {
            EXPECT_FLOAT_EQ(static_cast<float>(value), 1.f / 65536.f);
        }
    }

    TEST(CUDANNTest, directAndFusedActivationKernelsAgree) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            GTEST_SKIP() << "No CUDA device is available.";
        }
        using namespace ormi::core;
        const auto device = Device::from("cuda:0");
        const auto input = std::make_shared<Tensor>(device, Shape{7}, DType::float32);
        const std::array values{-1000.f, -2.f, -0.1f, 0.f, 0.1f, 2.f, 1000.f};
        runtime::memory::copy(device, input->data(), Device::cpu(), values.data(), sizeof(values));
        Tensor host_input(Device::cpu(), {7}, DType::float32);
        std::copy(values.begin(), values.end(), static_cast<float*>(host_input.data()));
        const std::array operations{
            ops::UnaryOp::tanh,
            ops::UnaryOp::sigmoid,
            ops::UnaryOp::softplus,
            ops::UnaryOp::relu,
            ops::UnaryOp::abs,
            ops::UnaryOp::sign,
            ops::UnaryOp::positive,
            ops::UnaryOp::erf,
            ops::UnaryOp::log1p,
            ops::UnaryOp::expm1,
            ops::UnaryOp::rsqrt
        };
        using Function = std::shared_ptr<Tensor> (*)(const std::shared_ptr<Tensor>&);
        const std::array<Function, 11> functions{
            ops::tanh,
            ops::sigmoid,
            ops::softplus,
            ops::relu,
            ops::abs,
            ops::sign,
            ops::positive,
            ops::erf,
            ops::log1p,
            ops::expm1,
            ops::rsqrt
        };
        for (std::size_t op = 0; op < operations.size(); ++op) {
            Tensor direct(device, {7}, DType::float32), host(Device::cpu(), {7}, DType::float32);
            runtime::pointwise::unary(operations[op], direct, *input);
            ormi::backends::cpu::pointwise_unary(operations[op], host, host_input);
            std::array<float, 7> actual{};
            runtime::memory::copy(Device::cpu(), actual.data(), device, direct.data(), sizeof(actual));
            const auto fused = functions[op](input);
            std::array<float, 7> optimized{};
            runtime::memory::copy(Device::cpu(), optimized.data(), device, fused->data(), sizeof(optimized));
            for (std::size_t i = 0; i < values.size(); ++i) {
                const auto expected = static_cast<const float*>(host.data())[i];
                if (std::isnan(expected)) {
                    EXPECT_TRUE(std::isnan(actual[i]));
                    EXPECT_TRUE(std::isnan(optimized[i]));
                } else if (std::isinf(expected)) {
                    EXPECT_EQ(actual[i], expected);
                    EXPECT_EQ(optimized[i], expected);
                } else {
                    EXPECT_NEAR(actual[i], expected, 1e-5f);
                    EXPECT_NEAR(optimized[i], actual[i], 1e-5f);
                }
            }
        }
    }
}
