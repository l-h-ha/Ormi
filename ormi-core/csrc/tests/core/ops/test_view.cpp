#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/attrs.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
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

    TEST(ViewTest, reshapeAliasesParentBuffer) {
        const auto input = floatTensor({2, 3});
        const auto result = ormi::core::ops::reshape(input, ormi::core::Shape{3, 2});

        EXPECT_FALSE(result->realized());
        ASSERT_NE(result->op(), nullptr);
        EXPECT_EQ(result->op()->name, "reshape");
        EXPECT_EQ(result->shape(), (ormi::core::Shape{3, 2}));
        EXPECT_EQ(result->strides()[0], 2);
        EXPECT_EQ(result->strides()[1], 1);

        result->realize();
        EXPECT_EQ(result->buffer(), input->buffer());
    }

    TEST(ViewTest, transposePermutesAxesAndParticipatesInCompute) {
        const auto input = floatTensor({2, 3});
        auto* input_data = static_cast<float*>(input->data());

        for (std::size_t i = 0; i < 6; ++i) {
            input_data[i] = static_cast<float>(i + 1);
        }

        const auto transposed = ormi::core::ops::transpose(
            input,
            std::vector<std::int64_t>{1, -2}
        );
        const auto& attrs = std::get<ormi::core::Permute>(transposed->attrs());
        const auto doubled = ormi::core::ops::add(transposed, transposed);
        const auto* data = static_cast<const float*>(doubled->data());

        EXPECT_EQ(transposed->shape(), (ormi::core::Shape{3, 2}));
        EXPECT_EQ(transposed->strides()[0], 1);
        EXPECT_EQ(transposed->strides()[1], 3);
        EXPECT_EQ(attrs.axes, (std::vector<std::int64_t>{1, 0}));
        EXPECT_FLOAT_EQ(data[0], 2.0f);
        EXPECT_FLOAT_EQ(data[1], 8.0f);
        EXPECT_FLOAT_EQ(data[2], 4.0f);
        EXPECT_FLOAT_EQ(data[3], 10.0f);
        EXPECT_FLOAT_EQ(data[4], 6.0f);
        EXPECT_FLOAT_EQ(data[5], 12.0f);
    }

    TEST(ViewTest, expandUsesZeroStridesAndParticipatesInCompute) {
        const auto input = floatTensor({2, 1});
        auto* input_data = static_cast<float*>(input->data());
        input_data[0] = 1.0f;
        input_data[1] = 2.0f;

        const auto expanded = ormi::core::ops::expand(input, ormi::core::Shape{2, 3});
        const auto doubled = ormi::core::ops::add(expanded, expanded);
        const auto* data = static_cast<const float*>(doubled->data());

        EXPECT_EQ(expanded->strides()[0], 1);
        EXPECT_EQ(expanded->strides()[1], 0);
        EXPECT_FLOAT_EQ(data[0], 2.0f);
        EXPECT_FLOAT_EQ(data[1], 2.0f);
        EXPECT_FLOAT_EQ(data[2], 2.0f);
        EXPECT_FLOAT_EQ(data[3], 4.0f);
        EXPECT_FLOAT_EQ(data[4], 4.0f);
        EXPECT_FLOAT_EQ(data[5], 4.0f);
    }

    TEST(ViewTest, reshapeInsertsSingletonAxesIntoExpandedViews) {
        const auto input = floatTensor({1});
        *static_cast<float*>(input->data()) = 2.0f;
        const auto expanded = ormi::core::ops::expand(input, ormi::core::Shape{3});
        const auto reshaped = ormi::core::ops::reshape(expanded, ormi::core::Shape{3, 1});
        const auto doubled = ormi::core::ops::add(reshaped, reshaped);
        const auto* data = static_cast<const float*>(doubled->data());

        EXPECT_EQ(reshaped->strides()[0], 0);
        EXPECT_EQ(reshaped->strides()[1], 0);
        EXPECT_FLOAT_EQ(data[0], 4.0f);
        EXPECT_FLOAT_EQ(data[1], 4.0f);
        EXPECT_FLOAT_EQ(data[2], 4.0f);
    }

    TEST(ViewTest, aliasRealizationRealizesLazyParent) {
        const auto lhs = floatTensor({2});
        const auto rhs = floatTensor({2});
        const auto parent = ormi::core::ops::add(lhs, rhs);
        const auto view = ormi::core::ops::reshape(parent, ormi::core::Shape{1, 2});

        EXPECT_FALSE(parent->realized());
        view->realize();
        EXPECT_TRUE(parent->realized());
        EXPECT_EQ(view->buffer(), parent->buffer());
    }

    TEST(ViewTest, rejectsInvalidViewShapesAndAxes) {
        const auto input = floatTensor({2, 3});

        EXPECT_THROW(
            (void)ormi::core::ops::reshape(input, ormi::core::Shape{5}),
            ormi::ShapeError
        );
        EXPECT_THROW(
            (void)ormi::core::ops::transpose(input, std::vector<std::int64_t>{0, 0}),
            ormi::ShapeError
        );
        EXPECT_THROW(
            (void)ormi::core::ops::expand(input, ormi::core::Shape{2, 4}),
            ormi::ShapeError
        );

        const auto transposed = ormi::core::ops::transpose(input, {1, 0});
        EXPECT_THROW(
            (void)ormi::core::ops::reshape(transposed, ormi::core::Shape{6}),
            ormi::ShapeError
        );
    }
}
