#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/linalg.h>
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

    class CUDALinalgTest : public ::testing::Test {
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

    TEST_F(CUDALinalgTest, realizesEveryVectorRankCombination) {
        const auto vector = tensor<float>({3}, {1.0f, 2.0f, 3.0f});
        const auto dot_rhs = tensor<float>({3}, {4.0f, 5.0f, 6.0f});
        auto dot = ormi::core::ops::matmul(vector, dot_rhs);
        EXPECT_EQ(dot->shape(), ormi::core::Shape{});
        EXPECT_FLOAT_EQ(host_values<float>(*dot)[0], 32.0f);

        const auto matrix = tensor<float>(
            {2, 3},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
        );
        auto matrix_vector = ormi::core::ops::matmul(matrix, vector);
        const auto matrix_vector_values = host_values<float>(*matrix_vector);
        EXPECT_EQ(matrix_vector->shape(), (ormi::core::Shape{2}));
        EXPECT_FLOAT_EQ(matrix_vector_values[0], 14.0f);
        EXPECT_FLOAT_EQ(matrix_vector_values[1], 32.0f);

        const auto rhs = tensor<float>(
            {3, 2},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
        );
        auto vector_matrix = ormi::core::ops::matmul(vector, rhs);
        const auto vector_matrix_values = host_values<float>(*vector_matrix);
        EXPECT_EQ(vector_matrix->shape(), (ormi::core::Shape{2}));
        EXPECT_FLOAT_EQ(vector_matrix_values[0], 22.0f);
        EXPECT_FLOAT_EQ(vector_matrix_values[1], 28.0f);
    }

    TEST_F(CUDALinalgTest, realizesBroadcastBatchesWithVectorOperands) {
        const auto lhs = tensor<float>(
            {2, 2, 3},
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
                10.0f, 11.0f, 12.0f
            }
        );
        const auto vector = tensor<float>({3}, {1.0f, 0.0f, -1.0f});
        auto result = ormi::core::ops::matmul(lhs, vector);
        const auto values = host_values<float>(*result);

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 2}));
        ASSERT_EQ(values.size(), 4u);
        for (const auto value : values) {
            EXPECT_FLOAT_EQ(value, -2.0f);
        }
    }

    TEST_F(CUDALinalgTest, realizesIrregularPartialBatchBroadcast) {
        const auto lhs = tensor<float>({2, 1, 1, 1}, {2.0f, 3.0f});
        const auto rhs = tensor<float>({3, 1, 1}, {10.0f, 20.0f, 30.0f});
        auto result = ormi::core::ops::matmul(lhs, rhs);
        const auto values = host_values<float>(*result);

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 3, 1, 1}));
        EXPECT_EQ(values, (std::vector<float>{20.0f, 40.0f, 60.0f, 30.0f, 60.0f, 90.0f}));
    }

    TEST_F(CUDALinalgTest, realizesStridedVectorFallback) {
        const auto storage = tensor<float>(
            {5},
            {1.0f, -100.0f, 2.0f, -100.0f, 3.0f}
        );
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            storage->buffer(),
            ormi::core::Shape{3},
            ormi::core::DType::float32,
            std::vector<std::int64_t>{2}
        );
        const auto rhs = tensor<float>({3}, {4.0f, 5.0f, 6.0f});
        auto result = ormi::core::ops::matmul(lhs, rhs);

        EXPECT_FLOAT_EQ(host_values<float>(*result)[0], 32.0f);
    }

    TEST_F(CUDALinalgTest, realizesFloat64Float16AndInt32) {
        auto float64_result = ormi::core::ops::matmul(
            tensor<double>({2}, {2.0, 3.0}),
            tensor<double>({2}, {4.0, 5.0})
        );
        EXPECT_DOUBLE_EQ(host_values<double>(*float64_result)[0], 23.0);

        auto float16_result = ormi::core::ops::matmul(
            tensor<ormi::core::Float16>(
                {2},
                {ormi::core::Float16{2.0f}, ormi::core::Float16{3.0f}}
            ),
            tensor<ormi::core::Float16>(
                {2},
                {ormi::core::Float16{4.0f}, ormi::core::Float16{5.0f}}
            )
        );
        EXPECT_FLOAT_EQ(
            static_cast<float>(host_values<ormi::core::Float16>(*float16_result)[0]),
            23.0f
        );

        auto int32_result = ormi::core::ops::matmul(
            tensor<std::int32_t>({2}, {2, 3}),
            tensor<std::int32_t>({2}, {4, 5})
        );
        EXPECT_EQ(host_values<std::int32_t>(*int32_result)[0], 23);
    }

    TEST_F(CUDALinalgTest, realizesDegenerateMatmulDimensions) {
        auto result = ormi::core::ops::matmul(
            tensor<float>({2, 0}, {}),
            tensor<float>({0, 3}, {})
        );
        const auto values = host_values<float>(*result);

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 3}));
        EXPECT_EQ(values, (std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));

        auto empty = ormi::core::ops::matmul(
            tensor<float>({0, 3}, {}),
            tensor<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
        );
        EXPECT_NO_THROW(empty->realize());
        EXPECT_EQ(empty->numel(), 0u);
        EXPECT_EQ(empty->data(), nullptr);
    }
}
