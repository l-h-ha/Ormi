#include <memory>
#include <variant>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> tensor(
        ormi::core::Shape shape,
        ormi::core::DType dtype = ormi::core::DType::float32
    ) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            dtype
        );
    }

    TEST(FusionExprTest, buildsFlatTypedDag) {
        const auto x = tensor({4});
        const auto square = ormi::core::ops::mul(x, x);
        const auto sum = ormi::core::ops::add(square, x);
        const auto exponential = ormi::core::ops::exp(sum);
        const auto output = ormi::core::ops::sin(exponential);
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        EXPECT_EQ(expression->dtype, ormi::core::DType::float32);
        EXPECT_EQ(expression->operations, 4u);
        ASSERT_EQ(expression->inputs.size(), 1u);
        EXPECT_EQ(expression->inputs[0], x.get());
        ASSERT_EQ(expression->nodes.size(), 5u);
        EXPECT_EQ(expression->output, 4u);

        const auto& input = std::get<ormi::core::runtime::fusion::Input>(expression->nodes[0]);
        const auto& multiply = std::get<ormi::core::runtime::fusion::Binary>(expression->nodes[1]);
        const auto& add = std::get<ormi::core::runtime::fusion::Binary>(expression->nodes[2]);
        const auto& exp = std::get<ormi::core::runtime::fusion::Unary>(expression->nodes[3]);
        const auto& sin = std::get<ormi::core::runtime::fusion::Unary>(expression->nodes[4]);

        EXPECT_EQ(input.index, 0u);
        EXPECT_EQ(multiply.op, ormi::core::ops::BinaryOp::mul);
        EXPECT_EQ(multiply.lhs, 0u);
        EXPECT_EQ(multiply.rhs, 0u);
        EXPECT_EQ(add.op, ormi::core::ops::BinaryOp::add);
        EXPECT_EQ(add.lhs, 1u);
        EXPECT_EQ(add.rhs, 0u);
        EXPECT_EQ(exp.op, ormi::core::ops::UnaryOp::exp);
        EXPECT_EQ(exp.input, 2u);
        EXPECT_EQ(sin.op, ormi::core::ops::UnaryOp::sin);
        EXPECT_EQ(sin.input, 3u);
    }

    TEST(FusionExprTest, representsConstantsWithoutInputs) {
        const auto x = tensor({4});
        const auto zero = ormi::core::ops::zeros_like(x);
        const auto output = ormi::core::ops::sin(ormi::core::ops::add(x, zero));
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        ASSERT_EQ(expression->inputs.size(), 1u);
        ASSERT_EQ(expression->nodes.size(), 4u);
        EXPECT_EQ(
            std::get<ormi::core::runtime::fusion::Constant>(expression->nodes[1]),
            ormi::core::runtime::fusion::Constant::zero
        );
    }

    TEST(FusionExprTest, requiresAtLeastTwoOperations) {
        const auto x = tensor({4});

        const auto output = ormi::core::ops::neg(x);
        EXPECT_FALSE(ormi::core::runtime::fusion::build(*output).has_value());
    }

    TEST(FusionExprTest, treatsRealizedOperationsAsInputs) {
        const auto x = tensor({4});
        const auto square = ormi::core::ops::mul(x, x);
        square->realize();
        const auto output = ormi::core::ops::sin(ormi::core::ops::add(square, x));
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        EXPECT_EQ(expression->operations, 2u);
        ASSERT_EQ(expression->inputs.size(), 2u);
        EXPECT_EQ(expression->inputs[0], square.get());
        EXPECT_EQ(expression->inputs[1], x.get());
    }

    TEST(FusionExprTest, rejectsBroadcastedOrNonContiguousInputs) {
        const auto lhs = tensor({2, 1});
        const auto rhs = tensor({1, 3});
        const auto broadcasted = ormi::core::ops::sin(ormi::core::ops::add(lhs, rhs));

        const auto base = tensor({2, 3});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto strided = ormi::core::ops::sin(ormi::core::ops::add(transposed, transposed));

        EXPECT_FALSE(ormi::core::runtime::fusion::build(*broadcasted).has_value());
        EXPECT_FALSE(ormi::core::runtime::fusion::build(*strided).has_value());
    }

    TEST(FusionExprTest, supportsFloat64) {
        const auto x = tensor({4}, ormi::core::DType::float64);
        const auto output = ormi::core::ops::sin(ormi::core::ops::mul(x, x));
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        EXPECT_EQ(expression->dtype, ormi::core::DType::float64);
    }
}
