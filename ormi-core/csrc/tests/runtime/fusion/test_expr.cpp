#include <array>
#include <memory>
#include <unordered_set>
#include <variant>
#include <gtest/gtest.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
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
        EXPECT_EQ(expression->shape, (std::vector<std::size_t>{4}));
        EXPECT_EQ(expression->operations, 4u);
        ASSERT_EQ(expression->inputs.size(), 1u);
        EXPECT_EQ(expression->inputs[0], x.get());
        ASSERT_EQ(expression->nodes.size(), 5u);
        EXPECT_EQ(expression->output, 4u);

        const auto& input = std::get<ormi::core::runtime::fusion::Input>(
            expression->nodes[0].instruction
        );
        const auto& multiply = std::get<ormi::core::runtime::fusion::Binary>(
            expression->nodes[1].instruction
        );
        const auto& add = std::get<ormi::core::runtime::fusion::Binary>(
            expression->nodes[2].instruction
        );
        const auto& exp = std::get<ormi::core::runtime::fusion::Unary>(
            expression->nodes[3].instruction
        );
        const auto& sin = std::get<ormi::core::runtime::fusion::Unary>(
            expression->nodes[4].instruction
        );

        EXPECT_EQ(input.index, 0u);
        EXPECT_TRUE(input.linear);
        EXPECT_TRUE(input.axes.empty());
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

    TEST(FusionExprTest, buildsSharedDagWithAuxiliaryOutput) {
        const auto x = tensor({4});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(x, x),
            x
        );
        const auto sine = ormi::core::ops::sin(mapped);
        const auto cosine = ormi::core::ops::cos(mapped);
        const std::array<ormi::core::Tensor*, 1> auxiliaries{cosine.get()};
        const std::unordered_set<const ormi::core::Tensor*> boundaries;

        const auto expression = ormi::core::runtime::fusion::build(
            *sine,
            auxiliaries,
            boundaries
        );

        ASSERT_TRUE(expression.has_value());
        EXPECT_EQ(expression->operations, 4u);
        EXPECT_EQ(expression->inputs, (std::vector<ormi::core::Tensor*>{x.get()}));
        EXPECT_EQ(expression->nodes.size(), 5u);
        EXPECT_EQ(expression->output, 3u);
        EXPECT_EQ(
            expression->auxiliary_outputs,
            (std::vector<ormi::core::runtime::fusion::Index>{4})
        );
    }

    TEST(FusionExprTest, rejectsDuplicateAuxiliaryOutput) {
        const auto x = tensor({4});
        const auto mapped = ormi::core::ops::add(x, x);
        const auto sine = ormi::core::ops::sin(mapped);
        const std::array<ormi::core::Tensor*, 1> auxiliaries{sine.get()};
        const std::unordered_set<const ormi::core::Tensor*> boundaries;

        EXPECT_FALSE(
            ormi::core::runtime::fusion::build(
                *sine,
                auxiliaries,
                boundaries
            ).has_value()
        );
    }

    TEST(FusionExprTest, appendsSharedParentsBeforeTheirConsumers) {
        const auto x = tensor({4});
        const auto y = tensor({4});
        const auto shared = ormi::core::ops::add(x, y);
        const auto left = ormi::core::ops::neg(shared);
        const auto right = ormi::core::ops::mul(shared, shared);
        const auto output = ormi::core::ops::sub(left, right);
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression);
        EXPECT_EQ(expression->inputs, (std::vector<ormi::core::Tensor*>{x.get(), y.get()}));
        EXPECT_EQ(expression->operations, 4u);
        EXPECT_EQ(expression->output, 5u);
        EXPECT_EQ(
            expression->nodes,
            (std::vector<ormi::core::runtime::fusion::Node>{
                {ormi::core::DType::float32, ormi::core::runtime::fusion::Input{0, {}, true}},
                {ormi::core::DType::float32, ormi::core::runtime::fusion::Input{1, {}, true}},
                {ormi::core::DType::float32, ormi::core::runtime::fusion::Binary{
                    ormi::core::ops::BinaryOp::add, 0, 1}},
                {ormi::core::DType::float32, ormi::core::runtime::fusion::Unary{
                    ormi::core::ops::UnaryOp::neg, 2}},
                {ormi::core::DType::float32, ormi::core::runtime::fusion::Binary{
                    ormi::core::ops::BinaryOp::mul, 2, 2}},
                {ormi::core::DType::float32, ormi::core::runtime::fusion::Binary{
                    ormi::core::ops::BinaryOp::sub, 3, 4}}
            })
        );
    }

    TEST(FusionExprTest, preservesAuxiliaryOutputOrderAcrossSharedParents) {
        const auto x = tensor({4});
        const auto y = tensor({4});
        const auto shared = ormi::core::ops::add(x, y);
        const auto first = ormi::core::ops::neg(shared);
        const auto second = ormi::core::ops::mul(shared, shared);
        const auto output = ormi::core::ops::sub(first, second);
        const std::array<ormi::core::Tensor*, 3> auxiliaries{
            second.get(), shared.get(), first.get()
        };
        const auto expression = ormi::core::runtime::fusion::build(*output, auxiliaries);

        ASSERT_TRUE(expression);
        EXPECT_EQ(expression->inputs, (std::vector<ormi::core::Tensor*>{x.get(), y.get()}));
        EXPECT_EQ(expression->operations, 4u);
        EXPECT_EQ(expression->nodes.size(), 6u);
        EXPECT_EQ(expression->output, 5u);
        EXPECT_EQ(
            expression->auxiliary_outputs,
            (std::vector<ormi::core::runtime::fusion::Index>{4, 2, 3})
        );
    }

    TEST(FusionExprTest, stopsAtExplicitOperationBoundary) {
        const auto x = tensor({4});
        const auto y = tensor({4});
        const auto shared = ormi::core::ops::add(x, y);
        const auto output = ormi::core::ops::sub(
            ormi::core::ops::neg(shared),
            ormi::core::ops::mul(shared, shared)
        );
        const std::unordered_set<const ormi::core::Tensor*> boundaries{shared.get()};
        const auto expression = ormi::core::runtime::fusion::build(*output, boundaries);

        ASSERT_TRUE(expression);
        EXPECT_FALSE(shared->realized());
        EXPECT_EQ(expression->inputs, (std::vector<ormi::core::Tensor*>{shared.get()}));
        EXPECT_EQ(expression->operations, 3u);
        ASSERT_EQ(expression->nodes.size(), 4u);
        EXPECT_EQ(expression->output, 3u);
        EXPECT_EQ(
            std::get<ormi::core::runtime::fusion::Input>(expression->nodes[0].instruction),
            (ormi::core::runtime::fusion::Input{0, {}, true})
        );
        EXPECT_EQ(
            std::get<ormi::core::runtime::fusion::Binary>(expression->nodes[2].instruction),
            (ormi::core::runtime::fusion::Binary{ormi::core::ops::BinaryOp::mul, 0, 0})
        );
    }

    TEST(FusionExprTest, recognizesConstantsBeforeExplicitBoundaries) {
        const auto x = tensor({4});
        const auto one = ormi::core::ops::ones_like(x);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(x, x), one)
        );
        const std::unordered_set<const ormi::core::Tensor*> boundaries{one.get()};
        const auto expression = ormi::core::runtime::fusion::build(*output, boundaries);

        ASSERT_TRUE(expression);
        EXPECT_EQ(expression->inputs, (std::vector<ormi::core::Tensor*>{x.get()}));
        EXPECT_EQ(expression->operations, 3u);
        ASSERT_EQ(expression->nodes.size(), 5u);
        EXPECT_EQ(expression->output, 4u);
        EXPECT_EQ(
            std::get<ormi::core::runtime::fusion::Constant>(expression->nodes[2].instruction),
            ormi::core::runtime::fusion::Constant::one
        );
    }

    TEST(FusionExprTest, representsConstantsWithoutInputs) {
        const auto x = tensor({4});
        const auto zero = ormi::core::ops::zeros_like(x);
        const auto output = ormi::core::ops::sin(ormi::core::ops::add(x, zero));
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        EXPECT_EQ(expression->shape, (std::vector<std::size_t>{4}));
        ASSERT_EQ(expression->inputs.size(), 1u);
        ASSERT_EQ(expression->nodes.size(), 4u);
        EXPECT_EQ(
            std::get<ormi::core::runtime::fusion::Constant>(
                expression->nodes[1].instruction
            ),
            ormi::core::runtime::fusion::Constant::zero
        );
    }

    TEST(FusionExprTest, requiresAtLeastTwoOperations) {
        const auto x = tensor({4});

        const auto output = ormi::core::ops::neg(x);
        EXPECT_FALSE(ormi::core::runtime::fusion::build(*output).has_value());
    }

    TEST(FusionExprTest, buildsLargeDag) {
        const auto input = tensor({4});
        auto output = input;

        constexpr std::size_t operations = 128;
        for (std::size_t operation = 0; operation < operations; ++operation) {
            output = ormi::core::ops::add(output, input);
        }

        const auto expression =
            ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        EXPECT_EQ(expression->operations, operations);
        EXPECT_EQ(expression->nodes.size(), operations + 1);
        EXPECT_EQ(expression->inputs.size(), 1u);
        EXPECT_EQ(expression->output, operations);
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

    TEST(FusionExprTest, recordsBroadcastInputAxes) {
        const auto lhs = tensor({2, 1});
        const auto rhs = tensor({1, 3});
        const auto broadcasted = ormi::core::ops::sin(ormi::core::ops::add(lhs, rhs));
        const auto expression = ormi::core::runtime::fusion::build(*broadcasted);

        ASSERT_TRUE(expression.has_value());
        ASSERT_EQ(expression->inputs.size(), 2u);

        const auto& lhs_input = std::get<ormi::core::runtime::fusion::Input>(
            expression->nodes[0].instruction
        );
        const auto& rhs_input = std::get<ormi::core::runtime::fusion::Input>(
            expression->nodes[1].instruction
        );

        EXPECT_FALSE(lhs_input.linear);
        ASSERT_EQ(lhs_input.axes.size(), 1u);
        EXPECT_EQ(lhs_input.axes[0], (ormi::core::runtime::fusion::Axis{3, 2, 1}));
        EXPECT_FALSE(rhs_input.linear);
        ASSERT_EQ(rhs_input.axes.size(), 1u);
        EXPECT_EQ(rhs_input.axes[0], (ormi::core::runtime::fusion::Axis{1, 3, 1}));
    }

    TEST(FusionExprTest, recordsScalarBroadcastWithoutAxes) {
        const auto lhs = tensor({2, 3});
        const auto rhs = tensor({});
        const auto output = ormi::core::ops::sin(ormi::core::ops::add(lhs, rhs));
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        ASSERT_EQ(expression->inputs.size(), 2u);
        const auto& rhs_input = std::get<ormi::core::runtime::fusion::Input>(
            expression->nodes[1].instruction
        );
        EXPECT_FALSE(rhs_input.linear);
        EXPECT_TRUE(rhs_input.axes.empty());
    }

    TEST(FusionExprTest, recordsNonContiguousInputAxes) {
        const auto base = tensor({2, 3});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto strided = ormi::core::ops::sin(ormi::core::ops::add(transposed, transposed));
        const auto expression = ormi::core::runtime::fusion::build(*strided);

        ASSERT_TRUE(expression.has_value());
        ASSERT_EQ(expression->inputs.size(), 1u);
        const auto& input = std::get<ormi::core::runtime::fusion::Input>(
            expression->nodes[0].instruction
        );
        EXPECT_FALSE(input.linear);
        ASSERT_EQ(input.axes.size(), 2u);
        EXPECT_EQ(input.axes[0], (ormi::core::runtime::fusion::Axis{2, 3, 1}));
        EXPECT_EQ(input.axes[1], (ormi::core::runtime::fusion::Axis{1, 2, 3}));
    }

    TEST(FusionExprTest, omitsExpandedAxesWithZeroStride) {
        const auto base = tensor({2, 1});
        const auto expanded = ormi::core::ops::expand(base, {2, 3});
        const auto output = ormi::core::ops::sin(ormi::core::ops::add(expanded, expanded));
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        const auto& input = std::get<ormi::core::runtime::fusion::Input>(
            expression->nodes[0].instruction
        );
        EXPECT_FALSE(input.linear);
        ASSERT_EQ(input.axes.size(), 1u);
        EXPECT_EQ(input.axes[0], (ormi::core::runtime::fusion::Axis{3, 2, 1}));
    }

    TEST(FusionExprTest, supportsFloat64) {
        const auto x = tensor({4}, ormi::core::DType::float64);
        const auto output = ormi::core::ops::sin(ormi::core::ops::mul(x, x));
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        EXPECT_EQ(expression->dtype, ormi::core::DType::float64);
    }

    TEST(FusionExprTest, supportsFloat16) {
        const auto x = tensor({4}, ormi::core::DType::float16);
        const auto output = ormi::core::ops::sin(ormi::core::ops::mul(x, x));
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        EXPECT_EQ(expression->dtype, ormi::core::DType::float16);
    }

    TEST(FusionExprTest, recordsAutocastAsTypedValues) {
        const auto x = tensor({4}, ormi::core::DType::float16);
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            output = ormi::core::ops::sum(ormi::core::ops::sin(x));
        }
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        ASSERT_EQ(expression->nodes.size(), 3u);
        EXPECT_EQ(expression->dtype, ormi::core::DType::float32);
        EXPECT_EQ(expression->nodes[0].dtype, ormi::core::DType::float16);
        EXPECT_EQ(expression->nodes[1].dtype, ormi::core::DType::float32);
        EXPECT_EQ(expression->nodes[2].dtype, ormi::core::DType::float32);

        const auto& cast = std::get<ormi::core::runtime::fusion::Cast>(
            expression->nodes[1].instruction
        );
        EXPECT_EQ(cast.input, 0u);
        EXPECT_EQ(
            std::get<ormi::core::runtime::fusion::Unary>(
                expression->nodes[2].instruction
            ).input,
            1u
        );
        ASSERT_TRUE(expression->reduction.has_value());
        EXPECT_EQ(expression->reduction->reduced.size, 4u);
    }

    TEST(FusionExprTest, recordsTerminalSum) {
        const auto x = tensor({2, 3});
        const auto mapped = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(x, x), x)
        );
        const auto output = ormi::core::ops::sum(mapped, true);
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        ASSERT_TRUE(expression->reduction.has_value());
        EXPECT_EQ(expression->elements, 6u);
        EXPECT_EQ(expression->reduction->op, ormi::core::ops::ReduceOp::sum);
        EXPECT_EQ(expression->reduction->output.size, 1u);
        EXPECT_TRUE(expression->reduction->output.axes.empty());
        EXPECT_EQ(expression->reduction->reduced.size, 6u);
        EXPECT_EQ(
            expression->reduction->reduced.axes,
            (std::vector<ormi::core::runtime::fusion::Axis>{
                {3, 2, 3},
                {1, 3, 1}
            })
        );
        EXPECT_EQ(expression->operations, 3u);
        EXPECT_EQ(expression->output, 3u);
    }

    TEST(FusionExprTest, recordsTerminalMean) {
        const auto x = tensor({2, 3});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(x, x),
            x
        );
        const auto output = ormi::core::ops::mean(mapped, {0});
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        ASSERT_TRUE(expression->reduction.has_value());
        EXPECT_EQ(expression->reduction->op, ormi::core::ops::ReduceOp::mean);
        EXPECT_EQ(expression->reduction->output.size, 3u);
        EXPECT_EQ(expression->reduction->reduced.size, 2u);
        EXPECT_EQ(expression->operations, 2u);
        EXPECT_EQ(expression->output, 2u);
    }

    TEST(FusionExprTest, keepdimsDoesNotChangeMappedReductionExpression) {
        const auto x = tensor({2, 3, 4});
        const auto mapped = ormi::core::ops::add(ormi::core::ops::mul(x, x), x);
        const auto squeezed = ormi::core::ops::sum(mapped, {0, 2}, false);
        const auto retained = ormi::core::ops::sum(mapped, {0, 2}, true);
        const auto first = ormi::core::runtime::fusion::build(*squeezed);
        const auto second = ormi::core::runtime::fusion::build(*retained);

        ASSERT_TRUE(first);
        ASSERT_TRUE(second);
        EXPECT_EQ(squeezed->shape(), (ormi::core::Shape{3}));
        EXPECT_EQ(retained->shape(), (ormi::core::Shape{1, 3, 1}));
        EXPECT_EQ(first->dtype, second->dtype);
        EXPECT_EQ(first->shape, second->shape);
        EXPECT_EQ(first->nodes, second->nodes);
        EXPECT_EQ(first->inputs, second->inputs);
        EXPECT_EQ(first->output, second->output);
        EXPECT_EQ(first->auxiliary_outputs, second->auxiliary_outputs);
        EXPECT_EQ(first->operations, second->operations);
        EXPECT_EQ(first->elements, second->elements);
        ASSERT_TRUE(first->reduction);
        EXPECT_EQ(first->reduction, second->reduction);
        EXPECT_EQ(first->reduction->output.size, 3u);
        EXPECT_EQ(first->reduction->reduced.size, 8u);
    }

    TEST(FusionExprTest, mapsPartialReductionCoordinatesToSource) {
        const auto x = tensor({2, 3, 4});
        const auto mapped = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(x, x), x)
        );
        const auto output = ormi::core::ops::sum(mapped, {1}, true);
        const auto expression = ormi::core::runtime::fusion::build(*output);

        ASSERT_TRUE(expression.has_value());
        ASSERT_TRUE(expression->reduction.has_value());
        EXPECT_EQ(expression->reduction->output.size, 8u);
        EXPECT_EQ(
            expression->reduction->output.axes,
            (std::vector<ormi::core::runtime::fusion::Axis>{
                {4, 2, 12},
                {1, 4, 1}
            })
        );
        EXPECT_EQ(expression->reduction->reduced.size, 3u);
        EXPECT_EQ(
            expression->reduction->reduced.axes,
            (std::vector<ormi::core::runtime::fusion::Axis>{{1, 3, 4}})
        );
    }
}
