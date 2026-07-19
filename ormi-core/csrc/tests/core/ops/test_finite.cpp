#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
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

    TEST(FiniteTest, constructsOneLazyScalarReduction) {
        const auto input = floatTensor({4});
        const auto result = ormi::core::ops::all_finite(input);

        EXPECT_FALSE(result->realized());
        EXPECT_TRUE(result->shape().scalar());
        EXPECT_EQ(result->dtype(), ormi::core::DType::boolean);
        ASSERT_NE(result->op(), nullptr);
        EXPECT_EQ(result->op()->name, "all_finite");
        EXPECT_EQ(result->op()->adjoint, nullptr);
        ASSERT_EQ(result->parents().size(), 1u);
        EXPECT_EQ(result->parents()[0], input);
    }

    TEST(CPUFiniteTest, detectsFloat32NanAndInfinity) {
        const auto input = floatTensor({4});
        auto* data = static_cast<float*>(input->data());
        data[0] = -2.0f;
        data[1] = 0.0f;
        data[2] = 3.0f;
        data[3] = 4.0f;

        EXPECT_TRUE(*static_cast<const bool*>(ormi::core::ops::all_finite(input)->data()));

        data[2] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(*static_cast<const bool*>(ormi::core::ops::all_finite(input)->data()));

        data[2] = std::numeric_limits<float>::infinity();
        EXPECT_FALSE(*static_cast<const bool*>(ormi::core::ops::all_finite(input)->data()));
    }

    TEST(CPUFiniteTest, supportsFloat64AndFloat16) {
        const auto input64 = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{2}, ormi::core::DType::float64
        );
        const auto input16 = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{2}, ormi::core::DType::float16
        );
        auto* data64 = static_cast<double*>(input64->data());
        auto* data16 = static_cast<ormi::core::Float16*>(input16->data());
        data64[0] = 1.0;
        data64[1] = -std::numeric_limits<double>::infinity();
        data16[0] = ormi::core::Float16(1.0f);
        data16[1] = ormi::core::Float16(2.0f);

        EXPECT_FALSE(*static_cast<const bool*>(ormi::core::ops::all_finite(input64)->data()));
        EXPECT_TRUE(*static_cast<const bool*>(ormi::core::ops::all_finite(input16)->data()));

        data16[1] = ormi::core::Float16(std::numeric_limits<float>::infinity());
        EXPECT_FALSE(*static_cast<const bool*>(ormi::core::ops::all_finite(input16)->data()));
    }

    TEST(CPUFiniteTest, readsNonContiguousInputAndAcceptsEmptyInput) {
        const auto input = floatTensor({2, 2});
        auto* data = static_cast<float*>(input->data());
        data[0] = 1.0f;
        data[1] = 2.0f;
        data[2] = std::numeric_limits<float>::quiet_NaN();
        data[3] = 4.0f;
        const auto transposed = ormi::core::ops::transpose(input, {1, 0});
        const auto empty = floatTensor({0});

        EXPECT_FALSE(*static_cast<const bool*>(ormi::core::ops::all_finite(transposed)->data()));
        EXPECT_TRUE(*static_cast<const bool*>(ormi::core::ops::all_finite(empty)->data()));
    }

    TEST(FiniteTest, rejectsNullAndNonFloatingInputs) {
        const auto integer = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{1}, ormi::core::DType::int32
        );
        const auto boolean = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{1}, ormi::core::DType::boolean
        );

        EXPECT_THROW((void)ormi::core::ops::all_finite(nullptr), ormi::OperationError);
        EXPECT_THROW((void)ormi::core::ops::all_finite(integer), ormi::DTypeError);
        EXPECT_THROW((void)ormi::core::ops::all_finite(boolean), ormi::DTypeError);
    }
}
