#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> floatTensor(ormi::core::Shape shape) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float32
        );
    }

    TEST(LinalgTest, matmulConstructsLazyTensor) {
        const auto lhs = floatTensor({2, 3});
        const auto rhs = floatTensor({3, 4});
        const auto result = ormi::core::ops::matmul(lhs, rhs);

        EXPECT_FALSE(result->realized());
        ASSERT_NE(result->op(), nullptr);
        EXPECT_EQ(result->op()->name, "matmul");
        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 4}));
        ASSERT_EQ(result->parents().size(), 2u);
        EXPECT_EQ(result->parents()[0], lhs);
        EXPECT_EQ(result->parents()[1], rhs);
    }

    TEST(LinalgTest, matmulInfersEverySupportedRankCombination) {
        using ormi::core::Shape;
        using ormi::core::ops::matmul_shape;

        EXPECT_EQ(matmul_shape(Shape{3}, Shape{3}).output, Shape{});
        EXPECT_EQ(matmul_shape(Shape{2, 3}, Shape{3}).output, (Shape{2}));
        EXPECT_EQ(matmul_shape(Shape{3}, Shape{3, 4}).output, (Shape{4}));
        EXPECT_EQ(matmul_shape(Shape{5, 2, 3}, Shape{3}).output, (Shape{5, 2}));
        EXPECT_EQ(matmul_shape(Shape{3}, Shape{5, 3, 4}).output, (Shape{5, 4}));
        EXPECT_EQ(
            matmul_shape(Shape{2, 1, 3, 4}, Shape{1, 3, 4, 5}).output,
            (Shape{2, 3, 3, 5})
        );
    }

    TEST(CPULinalgTest, realizesVectorDotProduct) {
        const auto lhs = floatTensor({3});
        const auto rhs = floatTensor({3});
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 1.0f;
        lhs_data[1] = 2.0f;
        lhs_data[2] = 3.0f;
        rhs_data[0] = 4.0f;
        rhs_data[1] = 5.0f;
        rhs_data[2] = 6.0f;

        const auto result = ormi::core::ops::matmul(lhs, rhs);

        EXPECT_EQ(result->shape(), ormi::core::Shape{});
        EXPECT_FLOAT_EQ(*static_cast<const float*>(result->data()), 32.0f);
    }

    TEST(CPULinalgTest, realizesMatrixVectorAndVectorMatrixProducts) {
        const auto matrix = floatTensor({2, 3});
        const auto vector = floatTensor({3});
        auto* matrix_data = static_cast<float*>(matrix->data());
        auto* vector_data = static_cast<float*>(vector->data());

        for (std::size_t i = 0; i < 6; ++i) {
            matrix_data[i] = static_cast<float>(i + 1);
        }
        vector_data[0] = 1.0f;
        vector_data[1] = 2.0f;
        vector_data[2] = 3.0f;

        const auto matrix_vector = ormi::core::ops::matmul(matrix, vector);
        const auto* matrix_vector_data = static_cast<const float*>(matrix_vector->data());
        EXPECT_EQ(matrix_vector->shape(), (ormi::core::Shape{2}));
        EXPECT_FLOAT_EQ(matrix_vector_data[0], 14.0f);
        EXPECT_FLOAT_EQ(matrix_vector_data[1], 32.0f);

        const auto rhs = floatTensor({3, 2});
        auto* rhs_data = static_cast<float*>(rhs->data());
        for (std::size_t i = 0; i < 6; ++i) {
            rhs_data[i] = static_cast<float>(i + 1);
        }

        const auto vector_matrix = ormi::core::ops::matmul(vector, rhs);
        const auto* vector_matrix_data = static_cast<const float*>(vector_matrix->data());
        EXPECT_EQ(vector_matrix->shape(), (ormi::core::Shape{2}));
        EXPECT_FLOAT_EQ(vector_matrix_data[0], 22.0f);
        EXPECT_FLOAT_EQ(vector_matrix_data[1], 28.0f);
    }

    TEST(CPULinalgTest, realizesBatchedProductsWithVectorOperands) {
        const auto lhs = floatTensor({2, 2, 3});
        const auto vector = floatTensor({3});
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* vector_data = static_cast<float*>(vector->data());

        for (std::size_t i = 0; i < 12; ++i) {
            lhs_data[i] = static_cast<float>(i + 1);
        }
        vector_data[0] = 1.0f;
        vector_data[1] = 0.0f;
        vector_data[2] = -1.0f;

        const auto result = ormi::core::ops::matmul(lhs, vector);
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 2}));
        EXPECT_FLOAT_EQ(data[0], -2.0f);
        EXPECT_FLOAT_EQ(data[1], -2.0f);
        EXPECT_FLOAT_EQ(data[2], -2.0f);
        EXPECT_FLOAT_EQ(data[3], -2.0f);
    }

    TEST(CPULinalgTest, realizesStridedVectorDotProduct) {
        const auto storage = floatTensor({5});
        auto* storage_data = static_cast<float*>(storage->data());
        storage_data[0] = 1.0f;
        storage_data[1] = -100.0f;
        storage_data[2] = 2.0f;
        storage_data[3] = -100.0f;
        storage_data[4] = 3.0f;
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            storage->buffer(),
            ormi::core::Shape{3},
            ormi::core::DType::float32,
            std::vector<std::int64_t>{2}
        );
        const auto rhs = floatTensor({3});
        auto* rhs_data = static_cast<float*>(rhs->data());
        rhs_data[0] = 4.0f;
        rhs_data[1] = 5.0f;
        rhs_data[2] = 6.0f;

        const auto result = ormi::core::ops::matmul(lhs, rhs);

        EXPECT_FLOAT_EQ(*static_cast<const float*>(result->data()), 32.0f);
    }

    TEST(CPULinalgTest, realizesDegenerateMatmulDimensions) {
        const auto lhs = floatTensor({2, 0});
        const auto rhs = floatTensor({0, 3});
        const auto result = ormi::core::ops::matmul(lhs, rhs);
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 3}));
        for (std::size_t i = 0; i < result->numel(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 0.0f);
        }

        const auto empty = ormi::core::ops::matmul(floatTensor({0, 3}), floatTensor({3, 2}));
        EXPECT_NO_THROW(empty->realize());
        EXPECT_EQ(empty->numel(), 0u);
        EXPECT_EQ(empty->data(), nullptr);
    }

    TEST(LinalgTest, realizesRankTwoMatmul) {
        const auto lhs = floatTensor({2, 3});
        const auto rhs = floatTensor({3, 2});
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());

        for (std::size_t i = 0; i < 6; ++i) {
            lhs_data[i] = static_cast<float>(i + 1);
            rhs_data[i] = static_cast<float>(i + 7);
        }

        const auto result = ormi::core::ops::matmul(lhs, rhs);
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_FLOAT_EQ(data[0], 58.0f);
        EXPECT_FLOAT_EQ(data[1], 64.0f);
        EXPECT_FLOAT_EQ(data[2], 139.0f);
        EXPECT_FLOAT_EQ(data[3], 154.0f);
    }

    TEST(LinalgTest, realizesFloat64Matmul) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float64
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float64
        );
        auto* lhs_data = static_cast<double*>(lhs->data());
        auto* rhs_data = static_cast<double*>(rhs->data());
        lhs_data[0] = 1.0;
        lhs_data[1] = 2.0;
        lhs_data[2] = 3.0;
        lhs_data[3] = 4.0;
        rhs_data[0] = 0.5;
        rhs_data[1] = 1.5;

        const auto result = ormi::core::ops::matmul(lhs, rhs);
        const auto* data = static_cast<const double*>(result->data());

        EXPECT_DOUBLE_EQ(data[0], 3.5);
        EXPECT_DOUBLE_EQ(data[1], 7.5);
    }

    TEST(CPULinalgTest, realizesFloat16MatmulWithFloat32Accumulation) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1, 4},
            ormi::core::DType::float16
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4, 1},
            ormi::core::DType::float16
        );
        auto* lhs_data = static_cast<ormi::core::Float16*>(lhs->data());
        auto* rhs_data = static_cast<ormi::core::Float16*>(rhs->data());
        const float lhs_values[] = {2048.0f, 1.0f, 1.0f, -2048.0f};

        for (std::size_t i = 0; i < 4; ++i) {
            lhs_data[i] = ormi::core::Float16(lhs_values[i]);
            rhs_data[i] = ormi::core::Float16(1.0f);
        }

        const auto result = ormi::core::ops::matmul(lhs, rhs);
        const auto value = *static_cast<const ormi::core::Float16*>(result->data());

        EXPECT_FLOAT_EQ(static_cast<float>(value), 2.0f);
    }

    TEST(CPULinalgTest, promotesMixedFloatingMatmul) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1, 2},
            ormi::core::DType::float16
        );
        const auto rhs = floatTensor({2, 1});
        auto* lhs_data = static_cast<ormi::core::Float16*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = ormi::core::Float16(1.0f);
        lhs_data[1] = ormi::core::Float16(2.0f);
        rhs_data[0] = 3.0f;
        rhs_data[1] = 4.0f;

        const auto result = ormi::core::ops::matmul(lhs, rhs);

        EXPECT_EQ(result->dtype(), ormi::core::DType::float32);
        ASSERT_NE(result->parents()[0]->op(), nullptr);
        EXPECT_EQ(result->parents()[0]->op()->name, "cast");
        EXPECT_EQ(result->parents()[1], rhs);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(result->data()), 11.0f);
    }

    TEST(LinalgTest, realizesBroadcastedBatchedMatmul) {
        const auto lhs = floatTensor({2, 2, 2});
        const auto rhs = floatTensor({1, 2, 1});
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        const float lhs_values[] = {
            1.0f, 0.0f,
            0.0f, 1.0f,
            2.0f, 0.0f,
            0.0f, 2.0f
        };

        for (std::size_t i = 0; i < 8; ++i) {
            lhs_data[i] = lhs_values[i];
        }
        rhs_data[0] = 3.0f;
        rhs_data[1] = 4.0f;

        const auto result = ormi::core::ops::matmul(lhs, rhs);
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 2, 1}));
        EXPECT_FLOAT_EQ(data[0], 3.0f);
        EXPECT_FLOAT_EQ(data[1], 4.0f);
        EXPECT_FLOAT_EQ(data[2], 6.0f);
        EXPECT_FLOAT_EQ(data[3], 8.0f);
    }

    TEST(LinalgTest, broadcastsDifferentBatchRanks) {
        const auto lhs = floatTensor({2, 1, 1, 1});
        const auto rhs = floatTensor({3, 1, 1});
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 2.0f;
        lhs_data[1] = 3.0f;
        rhs_data[0] = 10.0f;
        rhs_data[1] = 20.0f;
        rhs_data[2] = 30.0f;

        const auto result = ormi::core::ops::matmul(lhs, rhs);
        const auto* data = static_cast<const float*>(result->data());

        EXPECT_EQ(result->shape(), (ormi::core::Shape{2, 3, 1, 1}));
        EXPECT_FLOAT_EQ(data[0], 20.0f);
        EXPECT_FLOAT_EQ(data[1], 40.0f);
        EXPECT_FLOAT_EQ(data[2], 60.0f);
        EXPECT_FLOAT_EQ(data[3], 30.0f);
        EXPECT_FLOAT_EQ(data[4], 60.0f);
        EXPECT_FLOAT_EQ(data[5], 90.0f);
    }

    TEST(LinalgTest, rejectsInvalidMatmulShapes) {
        const auto matrix = floatTensor({3, 2});
        const auto wrong_inner = floatTensor({4, 2});
        const auto scalar = floatTensor({});

        EXPECT_THROW((void)ormi::core::ops::matmul(matrix, wrong_inner), ormi::ShapeError);
        EXPECT_THROW((void)ormi::core::ops::matmul(scalar, matrix), ormi::ShapeError);
    }

    TEST(LinalgTest, rejectsBooleanMatmulAtConstruction) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::boolean
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::boolean
        );

        EXPECT_THROW((void)ormi::core::ops::matmul(lhs, rhs), ormi::DTypeError);
    }

    TEST(LinalgTest, rejectsImplicitIntegerFloatingPromotion) {
        const auto lhs = floatTensor({2, 2});
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::int32
        );

        EXPECT_THROW((void)ormi::core::ops::matmul(lhs, rhs), ormi::DTypeError);
    }
}
