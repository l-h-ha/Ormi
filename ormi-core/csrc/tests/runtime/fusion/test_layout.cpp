#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/backend/fusion.h>
#include <ormi/core/shape.h>

#include "core/runtime/fusion/layout.h"

namespace {
    namespace fusion = ormi::core::runtime::fusion;
    namespace layout = fusion::layout;

    TEST(FusionLayoutTest, preservesScalarAndZeroDimensions) {
        EXPECT_TRUE(layout::dimensions({}).empty());
        EXPECT_EQ(layout::dimensions({2, 0, 4}), (std::vector<std::size_t>{2, 0, 4}));
    }

    TEST(FusionLayoutTest, mapsContiguousCoordinates) {
        std::vector<fusion::Axis> axes;
        ASSERT_TRUE(layout::input_axes({2, 3}, std::array<std::int64_t, 2>{3, 1}, {2, 3}, axes));
        EXPECT_EQ(axes, (std::vector<fusion::Axis>{{3, 2, 3}, {1, 3, 1}}));
    }

    TEST(FusionLayoutTest, scalarToScalarNeedsNoAxes) {
        std::vector<fusion::Axis> axes;
        ASSERT_TRUE(layout::input_axes({}, {}, {}, axes));
        EXPECT_TRUE(axes.empty());
    }

    TEST(FusionLayoutTest, broadcastsScalarWithoutAxes) {
        std::vector<fusion::Axis> axes;
        ASSERT_TRUE(layout::input_axes({}, {}, {2, 3, 4}, axes));
        EXPECT_TRUE(axes.empty());
    }

    TEST(FusionLayoutTest, alignsMixedRankBroadcastAxes) {
        std::vector<fusion::Axis> axes;
        ASSERT_TRUE(layout::input_axes({3, 1}, std::array<std::int64_t, 2>{1, 1}, {2, 3, 4}, axes));
        EXPECT_EQ(axes, (std::vector<fusion::Axis>{{4, 3, 1}}));
    }

    TEST(FusionLayoutTest, preservesTransposedStrides) {
        std::vector<fusion::Axis> axes;
        ASSERT_TRUE(layout::input_axes({3, 2}, std::array<std::int64_t, 2>{1, 3}, {3, 2}, axes));
        EXPECT_EQ(axes, (std::vector<fusion::Axis>{{2, 3, 1}, {1, 2, 3}}));
    }

    TEST(FusionLayoutTest, omitsExpandedZeroStrides) {
        std::vector<fusion::Axis> axes;
        ASSERT_TRUE(layout::input_axes({2, 3}, std::array<std::int64_t, 2>{1, 0}, {2, 3}, axes));
        EXPECT_EQ(axes, (std::vector<fusion::Axis>{{3, 2, 1}}));
    }

    TEST(FusionLayoutTest, emptyOutputDoesNotNeedInputCoordinates) {
        std::vector<fusion::Axis> axes;
        ASSERT_TRUE(layout::input_axes({3}, std::array<std::int64_t, 1>{1}, {0, 3}, axes));
        EXPECT_TRUE(axes.empty());
    }

    TEST(FusionLayoutTest, rejectsExcessRankOrIncompatibleExtents) {
        std::vector<fusion::Axis> axes;
        EXPECT_FALSE(layout::input_axes({1, 3}, std::array<std::int64_t, 2>{3, 1}, {3}, axes));
        EXPECT_FALSE(layout::input_axes({2, 4}, std::array<std::int64_t, 2>{4, 1}, {2, 3}, axes));
    }

    TEST(FusionLayoutTest, reusesAxesWithoutAccumulatingCoordinates) {
        std::vector<fusion::Axis> axes;
        ASSERT_TRUE(layout::input_axes({3, 1}, std::array<std::int64_t, 2>{1, 1}, {2, 3, 4}, axes));
        ASSERT_EQ(axes.size(), 1u);
        ASSERT_TRUE(layout::input_axes({}, {}, {2, 3, 4}, axes));
        EXPECT_TRUE(axes.empty());
    }

    TEST(FusionLayoutTest, partitionsSeparatedReductionAxes) {
        const auto reduction = layout::reduction(
            {2, 3, 4},
            std::array<std::int64_t, 2>{0, 2},
            ormi::core::ops::ReduceOp::mean
        );
        EXPECT_EQ(reduction.op, ormi::core::ops::ReduceOp::mean);
        EXPECT_EQ(reduction.output, (fusion::Iteration{3, {{1, 3, 4}}}));
        EXPECT_EQ(reduction.reduced, (fusion::Iteration{8, {{4, 2, 12}, {1, 4, 1}}}));
    }

    TEST(FusionLayoutTest, fullReductionHasOneOutputCoordinate) {
        const auto reduction = layout::reduction(
            {2, 3},
            std::array<std::int64_t, 2>{0, 1},
            ormi::core::ops::ReduceOp::sum
        );
        EXPECT_EQ(reduction.output, (fusion::Iteration{1, {}}));
        EXPECT_EQ(reduction.reduced, (fusion::Iteration{6, {{3, 2, 3}, {1, 3, 1}}}));
    }

    TEST(FusionLayoutTest, emptyReductionHasNoReducedCoordinates) {
        const auto reduction = layout::reduction(
            {2, 0, 4},
            std::array<std::int64_t, 1>{1},
            ormi::core::ops::ReduceOp::sum
        );
        EXPECT_EQ(reduction.output, (fusion::Iteration{8, {{4, 2, 0}, {1, 4, 1}}}));
        EXPECT_EQ(reduction.reduced, (fusion::Iteration{0, {}}));
    }

    TEST(FusionLayoutTest, emptyOutputHasNoOutputCoordinates) {
        const auto reduction = layout::reduction(
            {0, 3, 4},
            std::array<std::int64_t, 1>{1},
            ormi::core::ops::ReduceOp::mean
        );
        EXPECT_EQ(reduction.output, (fusion::Iteration{0, {}}));
        EXPECT_EQ(reduction.reduced, (fusion::Iteration{3, {{1, 3, 4}}}));
    }

    TEST(FusionLayoutTest, scalarReductionKeepsUnitIterations) {
        const auto reduction = layout::reduction({}, {}, ormi::core::ops::ReduceOp::sum);
        EXPECT_EQ(reduction.output, (fusion::Iteration{1, {}}));
        EXPECT_EQ(reduction.reduced, (fusion::Iteration{1, {}}));
    }
}
