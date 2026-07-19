#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    TEST(CreationTest, createsZerosAndOnesLike) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3},
            ormi::core::DType::float32
        );
        const auto zeros = ormi::core::ops::zeros_like(input);
        const auto ones = ormi::core::ops::ones_like(input);

        EXPECT_FALSE(zeros->realized());
        EXPECT_FALSE(ones->realized());
        EXPECT_TRUE(ormi::core::ops::is_zero(*zeros));
        EXPECT_TRUE(ormi::core::ops::is_one(*ones));
        EXPECT_TRUE(zeros->parents().empty());
        EXPECT_TRUE(ones->parents().empty());

        const auto* zero_data = static_cast<const float*>(zeros->data());
        const auto* one_data = static_cast<const float*>(ones->data());

        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_FLOAT_EQ(zero_data[i], 0.0f);
            EXPECT_FLOAT_EQ(one_data[i], 1.0f);
        }
    }

    TEST(CreationTest, createsFloat16OneBits) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float16
        );
        const auto ones = ormi::core::ops::ones_like(input);

        EXPECT_EQ(*static_cast<const std::uint16_t*>(ones->data()), 0x3c00);
    }

    TEST(CreationTest, createsFloat64Constants) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float64
        );
        const auto zeros = ormi::core::ops::zeros_like(input);
        const auto ones = ormi::core::ops::ones_like(input);
        const auto* zero_data = static_cast<const double*>(zeros->data());
        const auto* one_data = static_cast<const double*>(ones->data());

        EXPECT_DOUBLE_EQ(zero_data[0], 0.0);
        EXPECT_DOUBLE_EQ(zero_data[1], 0.0);
        EXPECT_DOUBLE_EQ(one_data[0], 1.0);
        EXPECT_DOUBLE_EQ(one_data[1], 1.0);
    }

    TEST(CreationTest, createsFullDouble) {
        const auto full = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{2.5},
            ormi::core::Device::cpu()
        );

        EXPECT_FALSE(full->realized());
        EXPECT_TRUE(full->parents().empty());
        EXPECT_EQ(full->shape().dims(), (std::vector<std::int64_t>{2, 3}));
        EXPECT_EQ(full->dtype(), ormi::core::DType::float64);

        const double* full_data = static_cast<const double*>(full->data());
        EXPECT_TRUE(full->realized());
        EXPECT_EQ(full_data[0], 2.5);
        EXPECT_EQ(full_data[1], 2.5);
        EXPECT_EQ(full_data[2], 2.5);
        EXPECT_EQ(full_data[3], 2.5);
        EXPECT_EQ(full_data[4], 2.5);
        EXPECT_EQ(full_data[5], 2.5);
    }

    TEST(CreationTest, createsFullFloat32) {
        const auto full = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{2.5f},
            ormi::core::Device::cpu()
        );

        EXPECT_FALSE(full->realized());
        EXPECT_TRUE(full->parents().empty());
        EXPECT_EQ(full->shape().dims(), (std::vector<std::int64_t>{2, 3}));
        EXPECT_EQ(full->dtype(), ormi::core::DType::float32);

        const float* full_data = static_cast<const float*>(full->data());
        EXPECT_TRUE(full->realized());
        EXPECT_EQ(full_data[0], 2.5f);
        EXPECT_EQ(full_data[1], 2.5f);
        EXPECT_EQ(full_data[2], 2.5f);
        EXPECT_EQ(full_data[3], 2.5f);
        EXPECT_EQ(full_data[4], 2.5f);
        EXPECT_EQ(full_data[5], 2.5f);
    }

    TEST(CreationTest, createsFullFloat16) {
        const auto val = ormi::core::Float16{2.5f};
        const auto full = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{val},
            ormi::core::Device::cpu()
        );

        EXPECT_FALSE(full->realized());
        EXPECT_TRUE(full->parents().empty());
        EXPECT_EQ(full->shape().dims(), (std::vector<std::int64_t>{2, 3}));
        EXPECT_EQ(full->dtype(), ormi::core::DType::float16);

        const auto* data = static_cast<const ormi::core::Float16*>(full->data());
        EXPECT_TRUE(full->realized());
        EXPECT_FLOAT_EQ(static_cast<float>(data[0]), 2.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(data[1]), 2.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(data[2]), 2.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(data[3]), 2.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(data[4]), 2.5f);
        EXPECT_FLOAT_EQ(static_cast<float>(data[5]), 2.5f);
    }

    TEST(CreationTest, createsFullInt32) {
        const auto full = ormi::core::ops::full(
            ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
            ormi::core::Scalar{std::int32_t{2}},
            ormi::core::Device::cpu()
        );

        EXPECT_FALSE(full->realized());
        EXPECT_TRUE(full->parents().empty());
        EXPECT_EQ(full->shape().dims(), (std::vector<std::int64_t>{2, 3}));
        EXPECT_EQ(full->dtype(), ormi::core::DType::int32);

        const auto* data = static_cast<const std::int32_t*>(full->data());
        EXPECT_TRUE(full->realized());
        EXPECT_EQ(data[0], 2);
        EXPECT_EQ(data[1], 2);
        EXPECT_EQ(data[2], 2);
        EXPECT_EQ(data[3], 2);
        EXPECT_EQ(data[4], 2);
        EXPECT_EQ(data[5], 2);
    }

    TEST(CreationTest, createsFullBoolean) {
        auto f = [](bool a) {
            const auto full = ormi::core::ops::full(
                ormi::core::Shape{std::vector<std::int64_t>{2, 3}},
                ormi::core::Scalar{a},
                ormi::core::Device::cpu()
            );

            EXPECT_FALSE(full->realized());
            EXPECT_TRUE(full->parents().empty());
            EXPECT_EQ(full->shape().dims(), (std::vector<std::int64_t>{2, 3}));
            EXPECT_EQ(full->dtype(), ormi::core::DType::boolean);

            const bool* full_data = static_cast<const bool*>(full->data());
            EXPECT_TRUE(full->realized());
            EXPECT_EQ(full_data[0], a);
            EXPECT_EQ(full_data[1], a);
            EXPECT_EQ(full_data[2], a);
            EXPECT_EQ(full_data[3], a);
            EXPECT_EQ(full_data[4], a);
            EXPECT_EQ(full_data[5], a);
        };

        f(true);
        f(false);
    }
}
