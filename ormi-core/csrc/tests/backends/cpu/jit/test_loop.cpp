#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/loop.h"
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

    ormi::backends::cpu::jit::Loop loop(
        const std::shared_ptr<ormi::core::Tensor>& lhs,
        const std::shared_ptr<ormi::core::Tensor>& rhs
    ) {
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto output = ormi::core::ops::add(square, rhs);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        EXPECT_TRUE(expression.has_value());
        return ormi::backends::cpu::jit::plan(*expression);
    }

    TEST(CPUJITLoopTest, keepsEqualInputsInOneFlatLoop) {
        const auto input = tensor({16});
        const auto result = loop(input, input);

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.span, 0u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear
        }));
    }

    TEST(CPUJITLoopTest, keepsScalarBroadcastInOneFlatLoop) {
        const auto result = loop(tensor({16}), tensor({}));

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.span, 0u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear,
            ormi::backends::cpu::jit::Access::scalar
        }));
    }

    TEST(CPUJITLoopTest, findsContiguousRowSpan) {
        const auto result = loop(tensor({2, 3}), tensor({3}));

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 3u);
        EXPECT_EQ(result.span, 3u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear,
            ormi::backends::cpu::jit::Access::span
        }));
    }

    TEST(CPUJITLoopTest, findsFixedColumnValue) {
        const auto result = loop(tensor({2, 3}), tensor({2, 1}));

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 3u);
        EXPECT_EQ(result.span, 3u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear,
            ormi::backends::cpu::jit::Access::fixed
        }));
    }

    TEST(CPUJITLoopTest, vectorizesExpandedColumnValue) {
        const auto column = tensor({1024, 1});
        const auto expanded = ormi::core::ops::expand(column, {1024, 32});
        const auto values = tensor({1024, 32});
        const auto multiplied = ormi::core::ops::mul(expanded, values);
        const auto output = ormi::core::ops::add(multiplied, values);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 32u);
        EXPECT_EQ(result.span, 32u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::fixed,
            ormi::backends::cpu::jit::Access::linear
        }));
    }

    TEST(CPUJITLoopTest, findsSpanAcrossArbitraryRank) {
        const auto result = loop(tensor({2, 3, 4}), tensor({3, 4}));

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 4u);
        EXPECT_EQ(result.span, 4u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear,
            ormi::backends::cpu::jit::Access::span
        }));
    }

    TEST(CPUJITLoopTest, combinesSpanAndFixedInputsAcrossArbitraryRank) {
        const auto result = loop(tensor({2, 1, 4}), tensor({1, 3, 1}));

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 4u);
        EXPECT_EQ(result.span, 4u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::span,
            ormi::backends::cpu::jit::Access::fixed
        }));
    }

    TEST(CPUJITLoopTest, plansTransposedInputAsStrided) {
        const auto base = tensor({2, 3});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto result = loop(transposed, transposed);

        EXPECT_FALSE(result.vectorized);
        EXPECT_TRUE(result.strided);
        EXPECT_FALSE(result.ranged);
        EXPECT_EQ(result.tile, 0u);
        EXPECT_EQ(result.strides, (std::vector<std::vector<std::size_t>>{
            {1, 3}
        }));
    }

    TEST(CPUJITLoopTest, alignsLinearAndTransposedInputStrides) {
        const auto base = tensor({2, 3});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto result = loop(transposed, tensor({3, 2}));

        EXPECT_TRUE(result.strided);
        EXPECT_EQ(result.strides, (std::vector<std::vector<std::size_t>>{
            {1, 3},
            {2, 1}
        }));
    }

    TEST(CPUJITLoopTest, tilesLargeTwoDimensionalTranspose) {
        const auto base = tensor({1024, 1024});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto result = loop(transposed, transposed);

        EXPECT_TRUE(result.strided);
        EXPECT_EQ(result.tile, 32u);
    }

    TEST(CPUJITLoopTest, vectorizesFloat32TranscendentalExpression) {
        const auto input = tensor({16});
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(input, input)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.math);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.lanes, 8u);
    }

    TEST(CPUJITLoopTest, vectorizesGlobalFloat32Reduction) {
        const auto input = tensor({64});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_FALSE(result.ranged);
        EXPECT_TRUE(result.partial);
        EXPECT_EQ(result.unit, 8u);
        EXPECT_EQ(result.lanes, 8u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear
        }));
    }

    TEST(CPUJITLoopTest, vectorizesLastAxisFloat64Reduction) {
        const auto input = tensor({2, 32}, ormi::core::DType::float64);
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.lanes, 4u);
    }

    TEST(CPUJITLoopTest, plansFirstAxisReductionWithDirectStrides) {
        const auto input = tensor({8, 4});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped, {0});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_FALSE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.strides, (std::vector<std::vector<std::size_t>>{
            {4, 1}
        }));
        EXPECT_EQ(result.output_dimensions, (std::vector<std::size_t>{1}));
        EXPECT_EQ(result.reduced_dimensions, (std::vector<std::size_t>{0}));
    }

    TEST(CPUJITLoopTest, plansMiddleAxisReductionAsOutputSlabs) {
        const auto input = tensor({2, 3, 4});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 4u);
        EXPECT_EQ(result.output_dimensions, (std::vector<std::size_t>{0, 2}));
        EXPECT_EQ(result.reduced_dimensions, (std::vector<std::size_t>{1}));
    }

    TEST(CPUJITLoopTest, vectorizesSeparatedAxesReduction) {
        const auto input = tensor({32, 16, 64});
        const auto row = tensor({64});
        const auto column = tensor({16, 1});
        const auto shifted = ormi::core::ops::add(
            ormi::core::ops::mul(input, row),
            column
        );
        const auto output = ormi::core::ops::mean(
            ormi::core::ops::mul(shifted, shifted),
            {0, 2}
        );
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.lanes, 8u);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.tile, 0u);
        EXPECT_EQ(result.strides, (std::vector<std::vector<std::size_t>>{
            {1024, 64, 1},
            {0, 0, 1},
            {0, 1, 0}
        }));
        EXPECT_EQ(result.output_dimensions, (std::vector<std::size_t>{1}));
        EXPECT_EQ(
            result.reduced_dimensions,
            (std::vector<std::size_t>{0, 2})
        );
    }

    TEST(CPUJITLoopTest, tilesLargeFirstAxisReduction) {
        const auto input = tensor({64, 70});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped, {0});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 32u);
        EXPECT_EQ(result.tile, 32u);
    }

    TEST(CPUJITLoopTest, tilesLargeTransposedLastAxisReduction) {
        const auto base = tensor({70, 33});
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            transposed
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 32u);
        EXPECT_EQ(result.tile, 32u);
        EXPECT_EQ(result.output_dimensions, (std::vector<std::size_t>{0}));
        EXPECT_EQ(result.reduced_dimensions, (std::vector<std::size_t>{1}));
    }

    TEST(CPUJITLoopTest, tilesRankThreeMiddleAxisReduction) {
        const auto base = tensor({64, 96, 32});
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(permuted, permuted),
            permuted
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.tile, 16u);
        EXPECT_EQ(result.tile_axis, 0u);
        EXPECT_EQ(result.unit, 16u * 96u);
        EXPECT_EQ(
            result.output_dimensions,
            (std::vector<std::size_t>{0, 2})
        );
        EXPECT_EQ(result.reduced_dimensions, (std::vector<std::size_t>{1}));
    }

    TEST(CPUJITLoopTest, leavesConflictingPartialReductionUntiled) {
        const auto base = tensor({64, 64});
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto contiguous = tensor({64, 64});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            contiguous
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_FALSE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.tile, 0u);
    }

    TEST(CPUJITLoopTest, plansTransposedReductionWithDirectStrides) {
        const auto base = tensor({2, 3});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto rhs = tensor({2});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            rhs
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_FALSE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.strides, (std::vector<std::vector<std::size_t>>{
            {1, 3},
            {0, 1}
        }));
        EXPECT_EQ(result.output_dimensions, (std::vector<std::size_t>{0}));
        EXPECT_EQ(result.reduced_dimensions, (std::vector<std::size_t>{1}));
    }

    TEST(CPUJITLoopTest, plansGlobalTransposeAsPartialSlabs) {
        const auto base = tensor({2, 3});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            transposed
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_TRUE(result.partial);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.split, 0u);
        EXPECT_TRUE(result.output_dimensions.empty());
        EXPECT_EQ(
            result.reduced_dimensions,
            (std::vector<std::size_t>{1, 0})
        );
    }

    TEST(CPUJITLoopTest, splitsLargestGlobalReducedDimension) {
        const auto base = tensor({1, 1});
        const auto expanded = ormi::core::ops::expand(base, {1, 1024});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(expanded, expanded),
            expanded
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.strided);
        EXPECT_TRUE(result.partial);
        EXPECT_TRUE(result.vectorized);
        EXPECT_EQ(result.lanes, 8u);
        EXPECT_EQ(result.unit, 8u);
        EXPECT_EQ(result.split, 1u);
        EXPECT_EQ(
            result.reduced_dimensions,
            (std::vector<std::size_t>{0, 1})
        );
    }

    TEST(CPUJITLoopTest, ordersSquareTransposeByPhysicalStride) {
        const auto base = tensor({1024, 1024});
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            transposed
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_EQ(result.split, 1u);
        EXPECT_TRUE(result.vectorized);
        EXPECT_EQ(result.lanes, 8u);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(
            result.reduced_dimensions,
            (std::vector<std::size_t>{1, 0})
        );
    }

    TEST(CPUJITLoopTest, vectorizesFloat64SquareTransposeWithFourLanes) {
        const auto base = tensor(
            {1025, 1025},
            ormi::core::DType::float64
        );
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            transposed
        );
        const auto output = ormi::core::ops::mean(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_EQ(result.lanes, 4u);
        EXPECT_EQ(result.split, 1u);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(
            result.reduced_dimensions,
            (std::vector<std::size_t>{1, 0})
        );
    }

    TEST(CPUJITLoopTest, tilesConflictingGlobalInputStrides) {
        const auto base = tensor({1024, 1024});
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto contiguous = tensor({1024, 1024});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            contiguous
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_EQ(result.split, 0u);
        EXPECT_FALSE(result.vectorized);
        EXPECT_EQ(result.tile, 16u);
        EXPECT_EQ(result.unit, 16u);
        EXPECT_EQ(
            result.reduced_dimensions,
            (std::vector<std::size_t>{0, 1})
        );
    }

    TEST(CPUJITLoopTest, tilesRankThreeConflictingGlobalInputStrides) {
        const auto base = tensor({64, 96, 32});
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto contiguous = tensor({32, 64, 96});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(permuted, permuted),
            contiguous
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.partial);
        EXPECT_FALSE(result.vectorized);
        EXPECT_EQ(result.split, 2u);
        EXPECT_EQ(result.tile_axis, 0u);
        EXPECT_EQ(result.tile, 16u);
        EXPECT_EQ(result.unit, 16u);
    }

    TEST(CPUJITLoopTest, leavesNarrowConflictingGlobalReductionUntiled) {
        const auto base = tensor({1024, 16});
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto contiguous = tensor({16, 1024});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            contiguous
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_FALSE(result.vectorized);
        EXPECT_EQ(result.tile, 0u);
        EXPECT_EQ(result.unit, 1u);
    }

    TEST(CPUJITLoopTest, leavesShortReductionScalar) {
        const auto input = tensor({2, 4});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_FALSE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
    }

    TEST(CPUJITLoopTest, leavesTranscendentalReductionScalar) {
        const auto input = tensor({16});
        const auto mapped = ormi::core::ops::sin(
            ormi::core::ops::add(input, input)
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        EXPECT_FALSE(ormi::backends::cpu::jit::plan(*expression).vectorized);
    }

    TEST(CPUJITLoopTest, vectorizesLargeFloat16Arithmetic) {
        const auto input = tensor({64}, ormi::core::DType::float16);
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::mul(mapped, input);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.lanes, 8u);
    }

    TEST(CPUJITLoopTest, vectorizesFloat16ScalarBroadcast) {
        const auto result = loop(
            tensor({64}, ormi::core::DType::float16),
            tensor({}, ormi::core::DType::float16)
        );

        EXPECT_TRUE(result.vectorized);
        EXPECT_EQ(result.lanes, 8u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear,
            ormi::backends::cpu::jit::Access::scalar
        }));
    }

    TEST(CPUJITLoopTest, vectorizesFloat16RowBroadcast) {
        const auto result = loop(
            tensor({2, 32}, ormi::core::DType::float16),
            tensor({32}, ormi::core::DType::float16)
        );

        EXPECT_TRUE(result.vectorized);
        EXPECT_EQ(result.span, 32u);
        EXPECT_EQ(result.lanes, 8u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear,
            ormi::backends::cpu::jit::Access::span
        }));
    }

    TEST(CPUJITLoopTest, vectorizesFloat16ColumnBroadcast) {
        const auto result = loop(
            tensor({2, 32}, ormi::core::DType::float16),
            tensor({2, 1}, ormi::core::DType::float16)
        );

        EXPECT_TRUE(result.vectorized);
        EXPECT_EQ(result.span, 32u);
        EXPECT_EQ(result.lanes, 8u);
        EXPECT_EQ(result.inputs, (std::vector{
            ormi::backends::cpu::jit::Access::linear,
            ormi::backends::cpu::jit::Access::fixed
        }));
    }

    TEST(CPUJITLoopTest, leavesNarrowFloat16BroadcastSpanScalar) {
        const auto result = loop(
            tensor({16, 4}, ormi::core::DType::float16),
            tensor({4}, ormi::core::DType::float16)
        );

        EXPECT_FALSE(result.vectorized);
    }

    TEST(CPUJITLoopTest, vectorizesGlobalFloat16Reduction) {
        const auto input = tensor({65}, ormi::core::DType::float16);
        const auto mapped = ormi::core::ops::mul(
            ormi::core::ops::add(
                ormi::core::ops::mul(input, input),
                input
            ),
            input
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_FALSE(result.ranged);
        EXPECT_TRUE(result.partial);
        EXPECT_EQ(result.unit, 8u);
        EXPECT_EQ(result.lanes, 8u);
    }

    TEST(CPUJITLoopTest, vectorizesLastAxisFloat16Reduction) {
        const auto input = tensor({2, 65}, ormi::core::DType::float16);
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.lanes, 8u);
    }

    TEST(CPUJITLoopTest, vectorizesAutocastFloat16ArithmeticReduction) {
        const auto input = tensor({65}, ormi::core::DType::float16);
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            const auto square = ormi::core::ops::mul(input, input);
            output = ormi::core::ops::sum(
                ormi::core::ops::add(square, input)
            );
        }
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto result = ormi::backends::cpu::jit::plan(*expression);

        EXPECT_TRUE(result.vectorized);
        EXPECT_EQ(result.lanes, 8u);
        EXPECT_EQ(output->dtype(), ormi::core::DType::float32);
    }

    TEST(CPUJITLoopTest, leavesAutocastFloat16TranscendentalReductionScalar) {
        const auto input = tensor({65}, ormi::core::DType::float16);
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            output = ormi::core::ops::sum(ormi::core::ops::sin(input));
        }
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        EXPECT_FALSE(ormi::backends::cpu::jit::plan(*expression).vectorized);
    }

    TEST(CPUJITLoopTest, leavesFirstAxisFloat16ReductionScalar) {
        const auto input = tensor({65, 2}, ormi::core::DType::float16);
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped, {0});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        EXPECT_FALSE(ormi::backends::cpu::jit::plan(*expression).vectorized);
    }

    TEST(CPUJITLoopTest, leavesShortFloat16ArithmeticScalar) {
        const auto input = tensor({63}, ormi::core::DType::float16);
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::mul(mapped, input);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        EXPECT_FALSE(ormi::backends::cpu::jit::plan(*expression).vectorized);
    }

    TEST(CPUJITLoopTest, leavesShortFloat16ReductionScalar) {
        const auto input = tensor({63}, ormi::core::DType::float16);
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        EXPECT_FALSE(ormi::backends::cpu::jit::plan(*expression).vectorized);
    }

    TEST(CPUJITLoopTest, preservesReductionVectorizationThresholds) {
        using namespace ormi::core;

        for (const auto dtype : {DType::float16, DType::float32, DType::float64}) {
            const std::int64_t width = dtype == DType::float64 ? 4 : 8;
            const auto minimum = width * 8;
            for (const auto size : {minimum - 1, minimum, minimum + 1}) {
                SCOPED_TRACE(static_cast<int>(dtype));
                SCOPED_TRACE(size);
                const auto input = tensor({size}, dtype);
                const auto output = ops::sum(ops::add(ops::mul(input, input), input));
                const auto expression = runtime::fusion::build(*output).value();
                const auto result = ormi::backends::cpu::jit::plan(expression);

                EXPECT_EQ(result.vectorized, size >= minimum);
                EXPECT_TRUE(result.partial);
                EXPECT_FALSE(result.ranged);
                EXPECT_EQ(result.unit, size >= minimum ? width : 1);
                EXPECT_EQ(result.lanes, size >= minimum ? width : 0);
            }
        }
    }

    TEST(CPUJITLoopTest, preservesOutputTileAndWorkerBoundaries) {
        using namespace ormi::core;

        for (const std::int64_t columns : {31, 32, 33}) {
            SCOPED_TRACE(columns);
            const auto input = tensor({64, columns});
            const auto output = ops::sum(ops::add(ops::mul(input, input), input), {0});
            const auto expression = runtime::fusion::build(*output).value();
            const auto result = ormi::backends::cpu::jit::plan(expression);

            EXPECT_TRUE(result.strided);
            EXPECT_EQ(result.vectorized, columns >= 32);
            EXPECT_EQ(result.tile, columns >= 32 ? 32u : 0u);
            EXPECT_EQ(result.ranged, columns != 32);
            EXPECT_EQ(result.unit, columns == 31 ? 1u : columns == 32 ? 0u : 32u);
        }
    }

    TEST(CPUJITLoopTest, keepsShortMathMapsRangedWithoutPacking) {
        using namespace ormi::core;

        for (const std::int64_t size : {7, 8}) {
            SCOPED_TRACE(size);
            const auto input = tensor({size});
            const auto output = ops::sin(ops::add(input, input));
            const auto expression = runtime::fusion::build(*output).value();
            const auto result = ormi::backends::cpu::jit::plan(expression);

            EXPECT_TRUE(result.ranged);
            EXPECT_EQ(result.unit, 1u);
            EXPECT_EQ(result.vectorized, size == 8);
            EXPECT_EQ(result.math, size == 8);
            EXPECT_EQ(result.lanes, size == 8 ? 8u : 0u);
        }
    }

    TEST(CPUJITLoopTest, keepsScalarMathBroadcastWorkerSpans) {
        using namespace ormi::core;
        using ormi::backends::cpu::jit::Access;

        for (const auto dtype : {DType::float32, DType::float64}) {
            const auto input = tensor({16, 4}, dtype);
            const auto row = tensor({4}, dtype);
            const auto output = ops::sin(ops::add(ops::mul(input, input), row));
            const auto expression = runtime::fusion::build(*output).value();
            const auto result = ormi::backends::cpu::jit::plan(expression);

            EXPECT_FALSE(result.vectorized);
            EXPECT_FALSE(result.math);
            EXPECT_TRUE(result.ranged);
            EXPECT_EQ(result.unit, 4u);
            EXPECT_EQ(result.span, 0u);
            EXPECT_EQ(result.inputs, (std::vector{Access::linear, Access::indexed}));
        }
    }

    TEST(CPUJITLoopTest, keepsEmptyMapsFlat) {
        const auto result = loop(tensor({0, 64}), tensor({64}));

        EXPECT_TRUE(result.vectorized);
        EXPECT_TRUE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.span, 0u);
    }

    TEST(CPUJITLoopTest, retainsInputClassificationBeforeFloat16Fallback) {
        using namespace ormi::core;
        using namespace runtime::fusion;
        using ormi::backends::cpu::jit::Access;

        const auto first = tensor({64}, DType::float16);
        const auto second = tensor({64}, DType::float16);
        const Expr expression{
            .dtype = DType::float16,
            .shape = {64},
            .nodes = {
                {DType::float16, Input{0, {}, true}},
                {DType::float64, Constant::one},
                {DType::float16, Input{2, {}, true}}
            },
            .inputs = {first.get(), second.get()},
            .output = 0,
            .operations = 1,
            .elements = 64
        };

        const auto result = ormi::backends::cpu::jit::plan(expression);
        EXPECT_FALSE(result.vectorized);
        EXPECT_FALSE(result.ranged);
        EXPECT_EQ(result.inputs, (std::vector{Access::linear, Access::indexed}));
    }

    TEST(CPUJITLoopTest, retainsReductionPartitionBeforeMixedDtypeFallback) {
        using namespace ormi::core;

        const auto input = tensor({64});
        const auto output = ops::sum(ops::add(ops::mul(input, input), input));
        auto expression = runtime::fusion::build(*output).value();
        expression.nodes.back().dtype = DType::float64;
        const auto result = ormi::backends::cpu::jit::plan(expression);

        EXPECT_FALSE(result.vectorized);
        EXPECT_TRUE(result.partial);
        EXPECT_FALSE(result.ranged);
        EXPECT_EQ(result.unit, 1u);
        EXPECT_EQ(result.inputs, (std::vector{ormi::backends::cpu::jit::Access::linear}));
    }

    TEST(CPUJITLoopTest, rejectsInvalidInputDuringContiguousAnalysis) {
        using namespace ormi::core;

        const auto input = tensor({64});
        const auto output = ops::add(ops::mul(input, input), input);
        auto expression = runtime::fusion::build(*output).value();
        std::get<runtime::fusion::Input>(expression.nodes.front().instruction).index = 1;

        EXPECT_THROW(ormi::backends::cpu::jit::plan(expression), ormi::OperationError);
    }

    TEST(CPUJITLoopTest, keepsInvalidStridedMetadataOnIndexedFallback) {
        using namespace ormi::core;
        using ormi::backends::cpu::jit::Access;

        const auto input = ops::transpose(tensor({32, 64}), {1, 0});
        const auto output = ops::add(ops::mul(input, input), input);
        auto expression = runtime::fusion::build(*output).value();
        std::get<runtime::fusion::Input>(expression.nodes.front().instruction).index = 1;
        const auto result = ormi::backends::cpu::jit::plan(expression);

        EXPECT_FALSE(result.strided);
        EXPECT_FALSE(result.vectorized);
        EXPECT_FALSE(result.ranged);
        EXPECT_TRUE(result.strides.empty());
        EXPECT_EQ(result.inputs, (std::vector{Access::indexed}));
    }

    TEST(CPUJITLoopTest, leavesNondividingBroadcastSpanIndexed) {
        using namespace ormi::core;
        using ormi::backends::cpu::jit::Access;

        const auto input = tensor({2, 4});
        const auto row = tensor({4});
        const auto output = ops::add(ops::mul(input, input), row);
        auto expression = runtime::fusion::build(*output).value();
        for (auto& node : expression.nodes) {
            auto* source = std::get_if<runtime::fusion::Input>(&node.instruction);
            if (source && !source->linear) {
                source->axes.front().extent = 3;
            }
        }
        const auto result = ormi::backends::cpu::jit::plan(expression);

        EXPECT_FALSE(result.vectorized);
        EXPECT_FALSE(result.ranged);
        EXPECT_EQ(result.span, 0u);
        EXPECT_EQ(result.inputs, (std::vector{Access::linear, Access::indexed}));
    }
}
