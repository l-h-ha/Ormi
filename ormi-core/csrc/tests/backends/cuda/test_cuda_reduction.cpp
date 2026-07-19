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
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
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

    template <>
    ormi::core::DType dtype<std::int32_t>() {
        return ormi::core::DType::int32;
    }

    class CUDAReductionTest : public ::testing::Test {
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

    TEST_F(CUDAReductionTest, realizesGlobalSumForEveryNumericDType) {
        auto float64 = ormi::core::ops::sum(
            tensor<double>({4}, {0.25, 0.5, 0.75, 1.0})
        );
        auto float32 = ormi::core::ops::sum(
            tensor<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
        );
        auto float16 = ormi::core::ops::sum(
            tensor<ormi::core::Float16>(
                {4},
                {
                    ormi::core::Float16{2048.0f},
                    ormi::core::Float16{1.0f},
                    ormi::core::Float16{1.0f},
                    ormi::core::Float16{-2048.0f}
                }
            )
        );
        auto int32 = ormi::core::ops::sum(
            tensor<std::int32_t>({4}, {1, 2, 3, 4})
        );

        EXPECT_DOUBLE_EQ(host_values<double>(*float64)[0], 2.5);
        EXPECT_FLOAT_EQ(host_values<float>(*float32)[0], 21.0f);
        EXPECT_FLOAT_EQ(
            static_cast<float>(host_values<ormi::core::Float16>(*float16)[0]),
            2.0f
        );
        EXPECT_EQ(host_values<std::int32_t>(*int32)[0], 10);
    }

    TEST_F(CUDAReductionTest, realizesMultipleAxesWithAndWithoutKeepdims) {
        const auto input = tensor<float>(
            {2, 3, 2},
            {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f
            }
        );
        auto squeezed = ormi::core::ops::sum(input, {0, 2});
        auto kept = ormi::core::ops::sum(input, {0, 2}, true);

        EXPECT_EQ(squeezed->shape(), (ormi::core::Shape{3}));
        EXPECT_EQ(kept->shape(), (ormi::core::Shape{1, 3, 1}));
        EXPECT_EQ(
            host_values<float>(*squeezed),
            (std::vector<float>{18.0f, 26.0f, 34.0f})
        );
        EXPECT_EQ(
            host_values<float>(*kept),
            (std::vector<float>{18.0f, 26.0f, 34.0f})
        );
    }

    TEST_F(CUDAReductionTest, realizesHierarchicalContiguousSum) {
        const auto input = ormi::core::ops::full(
            {2, 4096},
            1.0f,
            ormi::core::Device::cuda()
        );
        auto result = ormi::core::ops::sum(input, {1});

        EXPECT_EQ(
            host_values<float>(*result),
            (std::vector<float>{4096.0f, 4096.0f})
        );
    }

    TEST_F(CUDAReductionTest, realizesNonSuffixAndTransposedReductions) {
        const auto input = tensor<float>(
            {2, 3},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
        );
        auto first_axis = ormi::core::ops::sum(input, {0});
        auto transposed = ormi::core::ops::sum(
            ormi::core::ops::transpose(input, {1, 0}),
            {1}
        );

        EXPECT_EQ(
            host_values<float>(*first_axis),
            (std::vector<float>{5.0f, 7.0f, 9.0f})
        );
        EXPECT_EQ(
            host_values<float>(*transposed),
            (std::vector<float>{5.0f, 7.0f, 9.0f})
        );
    }

    TEST_F(CUDAReductionTest, writesZeroForEmptyReducedExtent) {
        const auto input = ormi::core::ops::full(
            {2, 0, 3},
            4.0f,
            ormi::core::Device::cuda()
        );
        auto result = ormi::core::ops::sum(input, {1});

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 3}));
        EXPECT_EQ(host_values<float>(*result), (std::vector<float>(6, 0.0f)));
    }

    TEST_F(CUDAReductionTest, realizesScalarAndEmptyOutput) {
        auto scalar = ormi::core::ops::sum(
            ormi::core::ops::full(
                ormi::core::Shape{},
                3.5f,
                ormi::core::Device::cuda()
            )
        );
        auto empty = ormi::core::ops::sum(
            ormi::core::ops::full(
                {0, 2, 3},
                1.0f,
                ormi::core::Device::cuda()
            ),
            {2}
        );

        EXPECT_FLOAT_EQ(host_values<float>(*scalar)[0], 3.5f);
        EXPECT_NO_THROW(empty->realize());
        EXPECT_TRUE(empty->realized());
        EXPECT_EQ(empty->shape(), (ormi::core::Shape{0, 2}));
        EXPECT_EQ(empty->data(), nullptr);
    }

    TEST_F(CUDAReductionTest, differentiatesReductionLazilyTwice) {
        const auto input = tensor<float>(ormi::core::Shape{}, {3.0f});
        const auto square = ormi::core::ops::mul(input, input);
        const auto total = ormi::core::ops::sum(square);
        auto second = ormi::core::diff(total, input, 2);

        EXPECT_FALSE(second->realized());
        EXPECT_FLOAT_EQ(host_values<float>(*second)[0], 2.0f);
    }

    TEST_F(CUDAReductionTest, limitsOnlyIndexedReductionRank) {
        std::vector<std::int64_t> dims(65, 1);
        dims[0] = 2;
        const auto expanded = ormi::core::ops::expand(
            ormi::core::ops::full(
                ormi::core::Shape{},
                2.0f,
                ormi::core::Device::cuda()
            ),
            ormi::core::Shape{dims}
        );
        auto indexed = ormi::core::ops::sum(expanded, {0});
        auto contiguous = ormi::core::ops::sum(
            ormi::core::ops::full(
                ormi::core::Shape{std::vector<std::int64_t>(65, 1)},
                2.0f,
                ormi::core::Device::cuda()
            )
        );

        EXPECT_FALSE(indexed->realized());
        EXPECT_THROW(indexed->realize(), ormi::ShapeError);
        EXPECT_FALSE(indexed->realized());
        EXPECT_FLOAT_EQ(host_values<float>(*contiguous)[0], 2.0f);
    }
}
