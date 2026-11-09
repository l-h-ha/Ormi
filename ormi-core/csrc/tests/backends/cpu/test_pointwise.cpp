#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/float16.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/pointwise.h"

namespace {
    using namespace ormi::core;
    using ops::BinaryOp;

    template <typename T>
    DType dtype() {
        if constexpr (std::is_same_v<T, float>) {
            return DType::float32;
        } else if constexpr (std::is_same_v<T, double>) {
            return DType::float64;
        } else if constexpr (std::is_same_v<T, Float16>) {
            return DType::float16;
        } else {
            return DType::int32;
        }
    }

    template <typename T>
    double value(T number) {
        if constexpr (std::is_same_v<T, Float16>) {
            return static_cast<float>(number);
        } else {
            return static_cast<double>(number);
        }
    }

    template <typename T>
    std::shared_ptr<Tensor> input(Shape shape) {
        auto result = std::make_shared<Tensor>(Device::cpu(), std::move(shape), dtype<T>());
        auto* values = static_cast<T*>(result->data());
        for (std::size_t i = 0; i < result->numel(); ++i) {
            values[i] = static_cast<T>(i % 17 + 1);
        }
        return result;
    }

    std::size_t reference_offset(std::size_t linear, const Shape& shape, const Tensor& input) {
        const auto leading = shape.rank() - input.rank();
        std::size_t offset = 0;
        for (std::size_t axis = shape.rank(); axis > 0; --axis) {
            const auto index = axis - 1;
            const auto coordinate = linear % static_cast<std::size_t>(shape[index]);
            linear /= static_cast<std::size_t>(shape[index]);
            if (index >= leading && input.shape()[index - leading] != 1) {
                offset += coordinate * static_cast<std::size_t>(input.strides()[index - leading]);
            }
        }
        return offset;
    }

    template <typename T>
    T reference(BinaryOp op, T lhs, T rhs) {
        switch (op) {
            case BinaryOp::add:
                return lhs + rhs;
            case BinaryOp::sub:
                return lhs - rhs;
            case BinaryOp::mul:
                return lhs * rhs;
            case BinaryOp::div:
                return lhs / rhs;
            case BinaryOp::minimum:
                return value(lhs) < value(rhs) ? lhs : rhs;
            case BinaryOp::maximum:
                return value(lhs) > value(rhs) ? lhs : rhs;
        }
        ADD_FAILURE() << "Unknown operation";
        return T{};
    }

    template <typename T>
    void check(const Tensor& lhs, const Tensor& rhs) {
        const auto shape = broadcast(lhs.shape(), rhs.shape());
        const auto count = shape.numel();
        auto storage = input<T>({static_cast<std::int64_t>(count + 2)});
        auto* guarded = static_cast<T*>(storage->data());
        guarded[0] = static_cast<T>(-1234);
        guarded[count + 1] = static_cast<T>(-2345);
        const auto first = guarded[0];
        const auto last = guarded[count + 1];
        Tensor output(storage->buffer(), shape, dtype<T>(), {}, sizeof(T));
        const auto* left = static_cast<const T*>(lhs.data());
        const auto* right = static_cast<const T*>(rhs.data());
        constexpr std::array operations{
            BinaryOp::add,
            BinaryOp::sub,
            BinaryOp::mul,
            BinaryOp::div,
            BinaryOp::minimum,
            BinaryOp::maximum
        };
        for (const auto op : operations) {
            SCOPED_TRACE(static_cast<int>(op));
            ormi::backends::cpu::pointwise_binary(op, output, lhs, rhs);
            EXPECT_EQ(value(guarded[0]), value(first));
            EXPECT_EQ(value(guarded[count + 1]), value(last));
            const auto* actual = static_cast<const T*>(output.data());
            for (std::size_t i = 0; i < count; ++i) {
                const auto expected = reference(
                    op,
                    left[reference_offset(i, shape, lhs)],
                    right[reference_offset(i, shape, rhs)]
                );
                ASSERT_EQ(value(actual[i]), value(expected)) << "index=" << i;
            }
        }
    }

    template <typename T>
    class CPUBroadcastTest : public testing::Test {};

    using NumericTypes = testing::Types<float, double, Float16, std::int32_t>;
    TYPED_TEST_SUITE(CPUBroadcastTest, NumericTypes);

    template <typename T>
    void check_unary(const Tensor& view) {
        const auto count = view.numel();
        const auto dense = input<T>(view.shape());
        auto* packed = static_cast<T*>(dense->data());
        const auto* source = static_cast<const T*>(view.data());
        for (std::size_t i = 0; i < count; ++i) {
            packed[i] = source[reference_offset(i, view.shape(), view)];
        }

        Tensor expected(Device::cpu(), view.shape(), dtype<T>());
        const auto storage = input<T>({static_cast<std::int64_t>(count + 2)});
        auto* guarded = static_cast<T*>(storage->data());
        guarded[0] = static_cast<T>(-1234);
        guarded[count + 1] = static_cast<T>(-2345);
        const auto first = guarded[0];
        const auto last = guarded[count + 1];
        Tensor output(storage->buffer(), view.shape(), dtype<T>(), {}, sizeof(T));
        constexpr std::array operations{
            ops::UnaryOp::neg, ops::UnaryOp::exp, ops::UnaryOp::log, ops::UnaryOp::sin,
            ops::UnaryOp::cos, ops::UnaryOp::abs, ops::UnaryOp::sign, ops::UnaryOp::positive,
            ops::UnaryOp::tanh, ops::UnaryOp::sigmoid, ops::UnaryOp::softplus, ops::UnaryOp::relu,
            ops::UnaryOp::erf, ops::UnaryOp::log1p, ops::UnaryOp::expm1, ops::UnaryOp::rsqrt
        };
        for (const auto op : operations) {
            if constexpr (std::is_same_v<T, std::int32_t>) {
                if (op != ops::UnaryOp::neg) {
                    continue;
                }
            }
            SCOPED_TRACE(static_cast<int>(op));
            ormi::backends::cpu::pointwise_unary(op, expected, *dense);
            ormi::backends::cpu::pointwise_unary(op, output, view);
            EXPECT_EQ(std::memcmp(&guarded[0], &first, sizeof(T)), 0);
            EXPECT_EQ(std::memcmp(&guarded[count + 1], &last, sizeof(T)), 0);
            const auto* actual = static_cast<const T*>(output.data());
            const auto* reference = static_cast<const T*>(expected.data());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_EQ(std::memcmp(&actual[i], &reference[i], sizeof(T)), 0) << "index=" << i;
            }
        }
    }

    template <typename T>
    class CPUUnaryTest : public testing::Test {};

    TYPED_TEST_SUITE(CPUUnaryTest, NumericTypes);

    TYPED_TEST(CPUUnaryTest, handlesDenseInputsAndScalarRanks) {
        check_unary<TypeParam>(*input<TypeParam>({}));
        check_unary<TypeParam>(*input<TypeParam>({1, 1, 1}));
        check_unary<TypeParam>(*input<TypeParam>({2, 3, 5}));
    }

    TYPED_TEST(CPUUnaryTest, walksConstantStrideSpansWithAnOffset) {
        const auto base = input<TypeParam>({70});
        const Tensor view(base->buffer(), {2, 3, 5}, dtype<TypeParam>(), {30, 10, 2}, 3 * sizeof(TypeParam));
        check_unary<TypeParam>(view);
    }

    TYPED_TEST(CPUUnaryTest, respectsRowGapsAndSingletonStrides) {
        const auto base = input<TypeParam>({64});
        const Tensor view(base->buffer(), {2, 1, 3, 1, 5, 1}, dtype<TypeParam>(),
            {23, 9999999, 7, 8888888, 1, 7777777}, 2 * sizeof(TypeParam));
        check_unary<TypeParam>(view);
        const Tensor columns(base->buffer(), {5, 2}, dtype<TypeParam>(), {9, 2});
        check_unary<TypeParam>(columns);
    }

    TYPED_TEST(CPUUnaryTest, readsPermutedAxes) {
        const auto base = input<TypeParam>({2, 3, 5});
        check_unary<TypeParam>(*ops::transpose(base, {1, 0, 2}));
        check_unary<TypeParam>(*ops::transpose(base, {2, 0, 1}));
    }

    TYPED_TEST(CPUUnaryTest, readsZeroStrideScalarsAndBlocks) {
        const Shape shape{2, 3, 5};
        check_unary<TypeParam>(*ops::expand(input<TypeParam>({}), shape));
        check_unary<TypeParam>(*ops::expand(input<TypeParam>({3, 1}), shape));
        const auto base = input<TypeParam>({2});
        const Tensor rows(base->buffer(), shape, dtype<TypeParam>(), {1, 0, 0});
        check_unary<TypeParam>(rows);
    }

    TYPED_TEST(CPUUnaryTest, handlesHighRankAndEmptyViews) {
        const Shape shape{2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
        auto strides = contiguous_strides(shape);
        for (std::size_t i = 0; i < strides.size(); ++i) {
            strides[i] = shape[i] == 1 ? 9999999 : strides[i] * 2;
        }
        const auto base = input<TypeParam>({132});
        const Tensor high(base->buffer(), shape, dtype<TypeParam>(), strides, 3 * sizeof(TypeParam));
        check_unary<TypeParam>(high);
        const auto empty = input<TypeParam>({0});
        const Tensor zero(empty->buffer(), {2, 0, 3}, dtype<TypeParam>(), {7, 99, 2});
        check_unary<TypeParam>(zero);
    }

    TYPED_TEST(CPUUnaryTest, preservesSequentialReadsWhenStorageOverlaps) {
        const auto base = input<TypeParam>({3});
        auto* values = static_cast<TypeParam*>(base->data());
        values[0] = static_cast<TypeParam>(1);
        const Tensor repeated(base->buffer(), {3}, dtype<TypeParam>(), {0});

        ormi::backends::cpu::pointwise_unary(ops::UnaryOp::neg, *base, repeated);

        EXPECT_EQ(value(values[0]), -1);
        EXPECT_EQ(value(values[1]), 1);
        EXPECT_EQ(value(values[2]), 1);
    }

    TYPED_TEST(CPUUnaryTest, preservesSequentialReadsAcrossOverlappingSpans) {
        const auto base = input<TypeParam>({8});
        Tensor output(base->buffer(), {2, 4}, dtype<TypeParam>());
        const Tensor repeated(base->buffer(), {2, 4}, dtype<TypeParam>(), {1, 0}, sizeof(TypeParam));

        ormi::backends::cpu::pointwise_unary(ops::UnaryOp::neg, output, repeated);

        const auto* actual = static_cast<const TypeParam*>(output.data());
        constexpr std::array expected{-2, -2, 2, 2, -2, -2, -2, -2};
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(value(actual[i]), expected[i]) << "index=" << i;
        }
    }

    TYPED_TEST(CPUUnaryTest, preservesSpecialValues) {
        if constexpr (!std::is_same_v<TypeParam, std::int32_t>) {
            const auto base = input<TypeParam>({12});
            auto* values = static_cast<TypeParam*>(base->data());
            values[0] = static_cast<TypeParam>(-0.0f);
            values[2] = static_cast<TypeParam>(0.0f);
            values[4] = static_cast<TypeParam>(std::numeric_limits<float>::infinity());
            values[6] = static_cast<TypeParam>(-std::numeric_limits<float>::infinity());
            values[8] = static_cast<TypeParam>(std::bit_cast<float>(std::uint32_t{0x7fc01234}));
            values[10] = static_cast<TypeParam>(-0.125f);
            const Tensor view(base->buffer(), {2, 3}, dtype<TypeParam>(), {6, 2});
            check_unary<TypeParam>(view);
            const Tensor repeated(base->buffer(), {2, 3, 17}, dtype<TypeParam>(), {6, 2, 0});
            check_unary<TypeParam>(repeated);
            for (std::size_t i = 0; i < 12; i += 2) {
                const Tensor scalar(base->buffer(), {37}, dtype<TypeParam>(), {0}, i * sizeof(TypeParam));
                check_unary<TypeParam>(scalar);
            }
        }
    }

    TYPED_TEST(CPUBroadcastTest, handlesDenseInputsAndScalarRanks) {
        const auto dense = input<TypeParam>({2, 3, 5});
        check<TypeParam>(*dense, *input<TypeParam>({2, 3, 5}));
        for (const auto& shape : {Shape{}, Shape{1}, Shape{1, 1, 1, 1}}) {
            const auto scalar = input<TypeParam>(shape);
            check<TypeParam>(*dense, *scalar);
            check<TypeParam>(*scalar, *dense);
        }
    }

    TYPED_TEST(CPUBroadcastTest, repeatsTrailingBlocksInEitherOperand) {
        const auto dense = input<TypeParam>({2, 3, 5});
        const auto block = input<TypeParam>({3, 5});
        check<TypeParam>(*dense, *block);
        check<TypeParam>(*block, *dense);
    }

    TYPED_TEST(CPUBroadcastTest, broadcastsAcrossSeveralAxes) {
        const auto lhs = input<TypeParam>({2, 1, 3, 1, 5});
        const auto rhs = input<TypeParam>({1, 7, 1, 2, 1});
        check<TypeParam>(*lhs, *rhs);
        check<TypeParam>(*rhs, *lhs);
        check<TypeParam>(*input<TypeParam>({2, 1}), *input<TypeParam>({1, 7}));
    }

    TYPED_TEST(CPUBroadcastTest, acceptsExpandedScalarsAndBlocks) {
        const auto shape = Shape{2, 3, 5};
        const auto scalar = ops::expand(input<TypeParam>({}), shape);
        const auto block = ops::expand(input<TypeParam>({3, 5}), shape);
        check<TypeParam>(*scalar, *scalar);
        check<TypeParam>(*scalar, *block);
        check<TypeParam>(*block, *input<TypeParam>(shape));
    }

    TYPED_TEST(CPUBroadcastTest, readsTransposedInputs) {
        const auto base = input<TypeParam>({2, 3, 5});
        const auto rows = ops::transpose(base, {1, 0, 2});
        const auto columns = ops::transpose(base, {2, 0, 1});
        check<TypeParam>(*rows, *input<TypeParam>({1, 2, 1}));
        check<TypeParam>(*columns, *input<TypeParam>({3}));
        check<TypeParam>(*input<TypeParam>({}), *columns);
    }

    TYPED_TEST(CPUBroadcastTest, respectsStridesAndStorageOffsets) {
        const auto base = input<TypeParam>({64});
        const Tensor
            strided(base->buffer(), {2, 3, 2}, dtype<TypeParam>(), {23, 5, 2}, 7 * sizeof(TypeParam));
        check<TypeParam>(strided, *input<TypeParam>({1, 3, 1}));
        check<TypeParam>(*input<TypeParam>({2, 3, 2}), strided);
        const Tensor singleton(base->buffer(), {2, 1, 3}, dtype<TypeParam>(), {6, 9999999, 2});
        check<TypeParam>(singleton, *input<TypeParam>({1, 1, 3}));
    }

    TYPED_TEST(CPUBroadcastTest, handlesHighRankAndEmptyOutputs) {
        std::vector<std::int64_t> left(12, 1), right(12, 1);
        for (std::size_t axis = 0; axis < left.size(); ++axis) {
            (axis % 2 == 0 ? left : right)[axis] = 2;
        }
        check<TypeParam>(*input<TypeParam>(Shape(left)), *input<TypeParam>(Shape(right)));
        check<TypeParam>(*input<TypeParam>({2, 0, 5}), *input<TypeParam>({1, 1, 5}));
        check<TypeParam>(*input<TypeParam>({}), *input<TypeParam>({0, 3}));
    }

    TEST(CPUBroadcastValuesTest, preservesNaNsAndSignedZeroInMinimumAndMaximum) {
        const auto lhs = input<float>({2, 3});
        auto* values = static_cast<float*>(lhs->data());
        values[0] = std::bit_cast<float>(std::uint32_t{0x7fc01234});
        values[1] = -0.0f;
        values[2] = 0.0f;
        values[3] = 2.0f;
        values[4] = -3.0f;
        values[5] = -0.0f;
        const auto scalar = input<float>({});
        *static_cast<float*>(scalar->data()) = 0.0f;
        Tensor output(Device::cpu(), {2, 3}, DType::float32);
        for (const auto op : {BinaryOp::minimum, BinaryOp::maximum}) {
            ormi::backends::cpu::pointwise_binary(op, output, *lhs, *scalar);
            const auto* result = static_cast<const float*>(output.data());
            EXPECT_EQ(std::bit_cast<std::uint32_t>(result[0]), std::uint32_t{0x7fc01234});
            EXPECT_FALSE(std::signbit(result[1]));
            EXPECT_FALSE(std::signbit(result[5]));
            ormi::backends::cpu::pointwise_binary(op, output, *scalar, *lhs);
            result = static_cast<const float*>(output.data());
            EXPECT_EQ(std::bit_cast<std::uint32_t>(result[0]), std::uint32_t{0x7fc01234});
            EXPECT_TRUE(std::signbit(result[1]));
            EXPECT_TRUE(std::signbit(result[5]));
        }
    }
}
