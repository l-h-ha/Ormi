#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/optim/grad_scaler.h>
#include <ormi/core/runtime/dispatch/finite.h>
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

    class CUDAFiniteTest : public ::testing::Test {
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

        static bool host_value(ormi::core::Tensor& tensor) {
            ormi::core::memory::Buffer host(
                ormi::core::Device::cpu(),
                sizeof(bool)
            );
            ormi::core::memory::copy(host, *tensor.buffer());
            return *static_cast<const bool*>(host.data());
        }
    };

    TEST_F(CUDAFiniteTest, publishesLazyBooleanOnCUDADevice) {
        const auto input = tensor<float>({4}, {-2.0f, 0.0f, 3.0f, 4.0f});
        auto result = ormi::core::ops::all_finite(input);

        EXPECT_FALSE(result->realized());
        EXPECT_TRUE(result->device().is_cuda());
        EXPECT_EQ(result->dtype(), ormi::core::DType::boolean);
        EXPECT_TRUE(result->shape().scalar());
        EXPECT_TRUE(host_value(*result));
    }

    TEST_F(CUDAFiniteTest, detectsNonFiniteValuesInEveryFloatingDType) {
        auto float64 = ormi::core::ops::all_finite(
            tensor<double>({2}, {1.0, std::numeric_limits<double>::infinity()})
        );
        auto float32 = ormi::core::ops::all_finite(
            tensor<float>({2}, {1.0f, std::numeric_limits<float>::quiet_NaN()})
        );
        auto float16 = ormi::core::ops::all_finite(
            tensor<ormi::core::Float16>(
                {2},
                {
                    ormi::core::Float16{1.0f},
                    ormi::core::Float16{std::numeric_limits<float>::infinity()}
                }
            )
        );

        EXPECT_FALSE(host_value(*float64));
        EXPECT_FALSE(host_value(*float32));
        EXPECT_FALSE(host_value(*float16));
    }

    TEST_F(CUDAFiniteTest, ordersLazyProducerBeforeScan) {
        const auto numerator = ormi::core::ops::full(
            {1024},
            1.0f,
            ormi::core::Device::cuda()
        );
        const auto denominator = ormi::core::ops::full(
            {1024},
            0.0f,
            ormi::core::Device::cuda()
        );
        const auto quotient = ormi::core::ops::div(numerator, denominator);
        auto result = ormi::core::ops::all_finite(quotient);

        EXPECT_FALSE(quotient->realized());
        EXPECT_FALSE(host_value(*result));
        EXPECT_TRUE(quotient->realized());
    }

    TEST_F(CUDAFiniteTest, scansTransposedAndEmptyInputs) {
        const auto input = tensor<float>(
            {2, 2},
            {
                1.0f,
                2.0f,
                std::numeric_limits<float>::quiet_NaN(),
                4.0f
            }
        );
        auto transposed = ormi::core::ops::all_finite(
            ormi::core::ops::transpose(input, {1, 0})
        );
        auto empty = ormi::core::ops::all_finite(
            ormi::core::ops::full({0}, 1.0f, ormi::core::Device::cuda())
        );

        EXPECT_FALSE(host_value(*transposed));
        EXPECT_TRUE(host_value(*empty));
    }

    TEST_F(CUDAFiniteTest, scansTensorBatchWithOneSynchronizationResult) {
        const auto first = tensor<float>({2}, {1.0f, 2.0f});
        const auto second = tensor<double>({2}, {3.0, 4.0});
        const std::array<const ormi::core::Tensor*, 2> finite{
            first.get(),
            second.get()
        };
        const auto invalid = tensor<ormi::core::Float16>(
            {2},
            {
                ormi::core::Float16{5.0f},
                ormi::core::Float16{std::numeric_limits<float>::quiet_NaN()}
            }
        );
        const std::array<const ormi::core::Tensor*, 3> mixed{
            first.get(),
            second.get(),
            invalid.get()
        };

        EXPECT_TRUE(ormi::core::runtime::finite::all(finite));
        EXPECT_FALSE(ormi::core::runtime::finite::all(mixed));
    }

    TEST_F(CUDAFiniteTest, drivesGradScalerOverflowDecision) {
        const std::vector<std::shared_ptr<ormi::core::Tensor>> parameters{
            tensor<float>(ormi::core::Shape{}, {3.0f}),
            tensor<float>(ormi::core::Shape{}, {5.0f})
        };
        const std::vector<std::shared_ptr<ormi::core::Tensor>> finite_gradients{
            tensor<float>(ormi::core::Shape{}, {48.0f}),
            tensor<float>(ormi::core::Shape{}, {16.0f})
        };
        const std::vector<std::shared_ptr<ormi::core::Tensor>> overflow_gradients{
            finite_gradients[0],
            tensor<float>(
                ormi::core::Shape{},
                {std::numeric_limits<float>::infinity()}
            )
        };
        ormi::core::optim::GradScaler scaler(8.0);

        const auto finite = scaler.step(parameters, finite_gradients, 0.1);
        ASSERT_TRUE(finite.has_value());
        EXPECT_FALSE((*finite)[0]->realized());
        EXPECT_FALSE((*finite)[1]->realized());

        EXPECT_FALSE(scaler.step(parameters, overflow_gradients, 0.1).has_value());
        EXPECT_DOUBLE_EQ(scaler.value(), 4.0);
    }

    TEST_F(CUDAFiniteTest, detectsFailureAtEndOfLargeInput) {
        std::vector<float> values(65537, 1.0f);
        values.back() = std::numeric_limits<float>::quiet_NaN();
        auto result = ormi::core::ops::all_finite(
            tensor<float>({static_cast<std::int64_t>(values.size())}, values)
        );

        EXPECT_FALSE(host_value(*result));
    }

    TEST_F(CUDAFiniteTest, limitsOnlyIndexedScanRank) {
        std::vector<std::int64_t> dims(65, 1);
        dims[0] = 2;
        const auto expanded = ormi::core::ops::expand(
            ormi::core::ops::full(
                ormi::core::Shape{},
                1.0f,
                ormi::core::Device::cuda()
            ),
            ormi::core::Shape{dims}
        );
        auto indexed = ormi::core::ops::all_finite(expanded);
        auto contiguous = ormi::core::ops::all_finite(
            ormi::core::ops::full(
                ormi::core::Shape{std::vector<std::int64_t>(65, 1)},
                1.0f,
                ormi::core::Device::cuda()
            )
        );

        EXPECT_FALSE(indexed->realized());
        EXPECT_THROW(indexed->realize(), ormi::ShapeError);
        EXPECT_FALSE(indexed->realized());
        EXPECT_TRUE(host_value(*contiguous));
    }
}
