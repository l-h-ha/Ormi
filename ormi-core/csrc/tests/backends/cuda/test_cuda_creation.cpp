#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    class CUDACreationTest : public ::testing::Test {
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

        template <typename T, std::size_t Size>
        static std::array<T, Size> copy_to_host(ormi::core::Tensor& tensor) {
            std::array<T, Size> output{};
            ormi::core::memory::Buffer host(
                ormi::core::Device::cpu(),
                output.size() * sizeof(T)
            );

            ormi::core::memory::copy(host, *tensor.buffer());
            std::memcpy(output.data(), host.data(), host.bytes());
            return output;
        }
    };

    TEST_F(CUDACreationTest, createsFloat64Full) {
        auto tensor = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{2.5},
            ormi::core::Device::cuda()
        );

        EXPECT_FALSE(tensor->realized());
        const auto output = copy_to_host<double, 6>(*tensor);
        EXPECT_TRUE(tensor->realized());

        for (const double value : output) {
            EXPECT_DOUBLE_EQ(value, 2.5);
        }
    }

    TEST_F(CUDACreationTest, createsFloat32Full) {
        auto tensor = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{2.5f},
            ormi::core::Device::cuda()
        );
        const auto output = copy_to_host<float, 6>(*tensor);

        for (const float value : output) {
            EXPECT_FLOAT_EQ(value, 2.5f);
        }
    }

    TEST_F(CUDACreationTest, createsFloat16Full) {
        const auto expected = ormi::core::Float16{2.5f};
        auto tensor = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{expected},
            ormi::core::Device::cuda()
        );
        const auto output = copy_to_host<ormi::core::Float16, 6>(*tensor);

        for (const auto value : output) {
            EXPECT_EQ(value.bits(), expected.bits());
        }
    }

    TEST_F(CUDACreationTest, createsInt32Full) {
        auto tensor = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{std::int32_t{7}},
            ormi::core::Device::cuda()
        );
        const auto output = copy_to_host<std::int32_t, 6>(*tensor);

        for (const std::int32_t value : output) {
            EXPECT_EQ(value, 7);
        }
    }

    TEST_F(CUDACreationTest, createsBooleanFull) {
        auto tensor = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{true},
            ormi::core::Device::cuda()
        );
        const auto output = copy_to_host<bool, 6>(*tensor);

        for (const bool value : output) {
            EXPECT_TRUE(value);
        }
    }

    TEST_F(CUDACreationTest, createsZerosAndOnesLike) {
        const auto input = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{4}},
            ormi::core::Scalar{2.0f},
            ormi::core::Device::cuda()
        );
        auto zeros = ormi::core::ops::zeros_like(input);
        auto ones = ormi::core::ops::ones_like(input);

        const auto zero_values = copy_to_host<float, 4>(*zeros);
        const auto one_values = copy_to_host<float, 4>(*ones);

        EXPECT_FALSE(input->realized());
        for (std::size_t i = 0; i < zero_values.size(); ++i) {
            EXPECT_FLOAT_EQ(zero_values[i], 0.0f);
            EXPECT_FLOAT_EQ(one_values[i], 1.0f);
        }
    }

    TEST_F(CUDACreationTest, createsEmptyFull) {
        auto tensor = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{0}},
            ormi::core::Scalar{2.5f},
            ormi::core::Device::cuda()
        );

        EXPECT_NO_THROW(tensor->realize());
        EXPECT_TRUE(tensor->realized());
        EXPECT_EQ(tensor->data(), nullptr);
    }

    TEST_F(CUDACreationTest, rejectsUnknownDType) {
        ormi::core::memory::Buffer buffer(ormi::core::Device::cuda(), sizeof(float));
        constexpr float value = 1.0f;

        EXPECT_THROW(
            ormi::core::memory::fill_scalar(
                buffer.data(),
                buffer.device(),
                &value,
                1,
                static_cast<ormi::core::DType>(255)
            ),
            ormi::DTypeError
        );
    }
}
