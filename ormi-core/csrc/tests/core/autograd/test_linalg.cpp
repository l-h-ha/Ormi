#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    TEST(AutogradLinalgTest, differentiatesVectorDotProduct) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3},
            ormi::core::DType::float32
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3},
            ormi::core::DType::float32
        );
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 1.0f;
        lhs_data[1] = 2.0f;
        lhs_data[2] = 3.0f;
        rhs_data[0] = 4.0f;
        rhs_data[1] = 5.0f;
        rhs_data[2] = 6.0f;

        const auto output = ormi::core::ops::matmul(lhs, rhs);
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );
        const auto* lhs_gradient = static_cast<const float*>(gradients[0]->data());
        const auto* rhs_gradient = static_cast<const float*>(gradients[1]->data());

        EXPECT_EQ(gradients[0]->shape(), lhs->shape());
        EXPECT_EQ(gradients[1]->shape(), rhs->shape());
        EXPECT_FLOAT_EQ(lhs_gradient[0], 4.0f);
        EXPECT_FLOAT_EQ(lhs_gradient[1], 5.0f);
        EXPECT_FLOAT_EQ(lhs_gradient[2], 6.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[0], 1.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[1], 2.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[2], 3.0f);
    }

    TEST(AutogradLinalgTest, differentiatesMatrixVectorProduct) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3},
            ormi::core::DType::float32
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3},
            ormi::core::DType::float32
        );
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        for (std::size_t i = 0; i < 6; ++i) {
            lhs_data[i] = static_cast<float>(i + 1);
        }
        rhs_data[0] = 1.0f;
        rhs_data[1] = 2.0f;
        rhs_data[2] = 3.0f;

        const auto output = ormi::core::ops::sum(ormi::core::ops::matmul(lhs, rhs));
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );
        const auto* lhs_gradient = static_cast<const float*>(gradients[0]->data());
        const auto* rhs_gradient = static_cast<const float*>(gradients[1]->data());

        for (std::size_t row = 0; row < 2; ++row) {
            EXPECT_FLOAT_EQ(lhs_gradient[row * 3], 1.0f);
            EXPECT_FLOAT_EQ(lhs_gradient[row * 3 + 1], 2.0f);
            EXPECT_FLOAT_EQ(lhs_gradient[row * 3 + 2], 3.0f);
        }
        EXPECT_FLOAT_EQ(rhs_gradient[0], 5.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[1], 7.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[2], 9.0f);
    }

    TEST(AutogradLinalgTest, reducesVectorGradientAcrossBatchedMatrices) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2, 1},
            ormi::core::DType::float32
        );
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 2.0f;
        lhs_data[1] = 3.0f;
        rhs_data[0] = 1.0f;
        rhs_data[1] = 2.0f;
        rhs_data[2] = 3.0f;
        rhs_data[3] = 4.0f;

        const auto output = ormi::core::ops::sum(ormi::core::ops::matmul(lhs, rhs));
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );
        const auto* lhs_gradient = static_cast<const float*>(gradients[0]->data());
        const auto* rhs_gradient = static_cast<const float*>(gradients[1]->data());

        EXPECT_FLOAT_EQ(lhs_gradient[0], 4.0f);
        EXPECT_FLOAT_EQ(lhs_gradient[1], 6.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[0], 2.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[1], 3.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[2], 2.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[3], 3.0f);
    }

    TEST(AutogradLinalgTest, differentiatesMatmul) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float32
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float32
        );
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 1.0f;
        lhs_data[1] = 2.0f;
        lhs_data[2] = 3.0f;
        lhs_data[3] = 4.0f;
        rhs_data[0] = 5.0f;
        rhs_data[1] = 6.0f;

        const auto product = ormi::core::ops::matmul(lhs, rhs);
        const auto output = ormi::core::ops::sum(product);
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );
        const auto* lhs_gradient = static_cast<const float*>(gradients[0]->data());
        const auto* rhs_gradient = static_cast<const float*>(gradients[1]->data());

        EXPECT_FLOAT_EQ(lhs_gradient[0], 5.0f);
        EXPECT_FLOAT_EQ(lhs_gradient[1], 6.0f);
        EXPECT_FLOAT_EQ(lhs_gradient[2], 5.0f);
        EXPECT_FLOAT_EQ(lhs_gradient[3], 6.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[0], 4.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[1], 6.0f);
    }

    TEST(AutogradLinalgTest, differentiatesFloat64Matmul) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1, 2},
            ormi::core::DType::float64
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float64
        );
        auto* lhs_data = static_cast<double*>(lhs->data());
        auto* rhs_data = static_cast<double*>(rhs->data());
        lhs_data[0] = 2.0;
        lhs_data[1] = 3.0;
        rhs_data[0] = 4.0;
        rhs_data[1] = 5.0;

        const auto output = ormi::core::ops::sum(ormi::core::ops::matmul(lhs, rhs));
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );
        const auto* lhs_gradient = static_cast<const double*>(gradients[0]->data());
        const auto* rhs_gradient = static_cast<const double*>(gradients[1]->data());

        EXPECT_DOUBLE_EQ(lhs_gradient[0], 4.0);
        EXPECT_DOUBLE_EQ(lhs_gradient[1], 5.0);
        EXPECT_DOUBLE_EQ(rhs_gradient[0], 2.0);
        EXPECT_DOUBLE_EQ(rhs_gradient[1], 3.0);
    }

    TEST(CPUAutogradLinalgTest, returnsPromotedMatmulGradientsToSourceDTypes) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1, 2},
            ormi::core::DType::float16
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float32
        );
        auto* lhs_data = static_cast<ormi::core::Float16*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = ormi::core::Float16(1.0f);
        lhs_data[1] = ormi::core::Float16(2.0f);
        rhs_data[0] = 3.0f;
        rhs_data[1] = 4.0f;

        const auto output = ormi::core::ops::sum(ormi::core::ops::matmul(lhs, rhs));
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );

        EXPECT_EQ(gradients[0]->dtype(), ormi::core::DType::float16);
        EXPECT_EQ(gradients[1]->dtype(), ormi::core::DType::float32);
        const auto* lhs_gradient = static_cast<const ormi::core::Float16*>(
            gradients[0]->data()
        );
        const auto* rhs_gradient = static_cast<const float*>(gradients[1]->data());
        EXPECT_FLOAT_EQ(static_cast<float>(lhs_gradient[0]), 3.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(lhs_gradient[1]), 4.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[0], 1.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[1], 2.0f);
    }

    TEST(AutogradLinalgTest, reversesBatchedMatmulBroadcasting) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2, 2},
            ormi::core::DType::float32
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1, 2, 1},
            ormi::core::DType::float32
        );
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

        const auto output = ormi::core::ops::sum(ormi::core::ops::matmul(lhs, rhs));
        const auto gradients = ormi::core::grad(
            output,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{lhs, rhs}
        );
        const auto* lhs_gradient = static_cast<const float*>(gradients[0]->data());
        const auto* rhs_gradient = static_cast<const float*>(gradients[1]->data());

        for (std::size_t batch = 0; batch < 2; ++batch) {
            EXPECT_FLOAT_EQ(lhs_gradient[batch * 4], 3.0f);
            EXPECT_FLOAT_EQ(lhs_gradient[batch * 4 + 1], 4.0f);
            EXPECT_FLOAT_EQ(lhs_gradient[batch * 4 + 2], 3.0f);
            EXPECT_FLOAT_EQ(lhs_gradient[batch * 4 + 3], 4.0f);
        }
        EXPECT_FLOAT_EQ(rhs_gradient[0], 3.0f);
        EXPECT_FLOAT_EQ(rhs_gradient[1], 3.0f);
    }
}
