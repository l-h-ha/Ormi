#include <limits>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/core/shape.h>

namespace {
    TEST(ShapeTest, scalarShapeHasOneElement) {
        const ormi::core::Shape shape;

        EXPECT_TRUE(shape.scalar());
        EXPECT_EQ(shape.rank(), 0u);
        EXPECT_EQ(shape.numel(), 1u);
    }

    TEST(ShapeTest, computesElementCount) {
        const ormi::core::Shape shape{2, 3, 4};

        EXPECT_EQ(shape.rank(), 3u);
        EXPECT_EQ(shape.numel(), 24u);
        EXPECT_FALSE(shape.empty());
    }

    TEST(ShapeTest, rejectsOverflowingElementCountWithoutZeroDimensions) {
        const auto limit = std::numeric_limits<std::int64_t>::max();
        const ormi::core::Shape shape{limit, limit};

        EXPECT_THROW((void)shape.numel(), ormi::ShapeError);
    }

    TEST(ShapeTest, zeroDimensionMakesShapeEmpty) {
        const ormi::core::Shape shape{2, 0, 4};

        EXPECT_EQ(shape.numel(), 0u);
        EXPECT_TRUE(shape.empty());
    }

    TEST(ShapeTest, zeroDimensionAfterLargeDimensionsStillMakesShapeEmpty) {
        const auto limit = std::numeric_limits<std::int64_t>::max();
        const ormi::core::Shape shape{limit, limit, 0};

        EXPECT_EQ(shape.numel(), 0u);
        EXPECT_TRUE(shape.empty());
    }

    TEST(ShapeTest, rejectsNegativeDimensions) {
        EXPECT_THROW((void)ormi::core::Shape({2, -1}), ormi::ShapeError);
    }

    TEST(ShapeTest, computesContiguousStrides) {
        const auto strides = ormi::core::contiguous_strides(ormi::core::Shape{2, 3, 4});

        ASSERT_EQ(strides.size(), 3u);
        EXPECT_EQ(strides[0], 12);
        EXPECT_EQ(strides[1], 4);
        EXPECT_EQ(strides[2], 1);
    }

    TEST(ShapeTest, detectsContiguousStrides) {
        const ormi::core::Shape shape{2, 3, 4};

        EXPECT_TRUE(ormi::core::is_contiguous(shape, std::vector<std::int64_t>{12, 4, 1}));
        EXPECT_FALSE(ormi::core::is_contiguous(shape, std::vector<std::int64_t>{1, 4, 12}));
    }

    TEST(ShapeTest, rejectsOverflowingContiguousStrides) {
        const auto limit = std::numeric_limits<std::int64_t>::max();

        EXPECT_THROW((void)ormi::core::contiguous_strides({0, limit, 2}), ormi::ShapeError);
        EXPECT_THROW((void)ormi::core::contiguous_strides({1, limit, 2}), ormi::ShapeError);
    }

    TEST(ShapeTest, acceptsLargestRepresentableContiguousStrides) {
        const auto limit = std::numeric_limits<std::int64_t>::max();
        const ormi::core::Shape shape{0, 1, limit};
        const auto strides = ormi::core::contiguous_strides(shape);

        EXPECT_EQ(strides, (std::vector<std::int64_t>{limit, limit, 1}));
        EXPECT_TRUE(ormi::core::is_contiguous(shape, strides));
        EXPECT_EQ(ormi::core::contiguous_strides({0, 2, limit/2}),
                  (std::vector<std::int64_t>{limit - 1, limit/2, 1}));
    }

    TEST(ShapeTest, contiguityDoesNotMultiplyTheLeadingDimension) {
        const auto limit = std::numeric_limits<std::int64_t>::max();

        EXPECT_TRUE(ormi::core::is_contiguous({limit}, std::vector<std::int64_t>{1}));
        EXPECT_TRUE(ormi::core::is_contiguous({2, limit}, std::vector<std::int64_t>{limit, 1}));
        EXPECT_TRUE(ormi::core::is_contiguous({limit, 2}, std::vector<std::int64_t>{2, 1}));
    }

    TEST(ShapeTest, contiguityRejectsUnrepresentableStrides) {
        const auto limit = std::numeric_limits<std::int64_t>::max();

        EXPECT_FALSE(ormi::core::is_contiguous({0, limit, 2}, std::vector<std::int64_t>{0, 2, 1}));
    }

    TEST(ShapeTest, trailingZeroPreventsStrideOverflow) {
        const auto limit = std::numeric_limits<std::int64_t>::max();
        const ormi::core::Shape shape{limit, limit, 0};
        const auto strides = ormi::core::contiguous_strides(shape);

        EXPECT_EQ(strides, (std::vector<std::int64_t>{0, 0, 1}));
        EXPECT_TRUE(ormi::core::is_contiguous(shape, strides));
    }

    TEST(ShapeTest, broadcastsShapes) {
        const auto shape = ormi::core::broadcast(
            ormi::core::Shape{2, 1, 4},
            ormi::core::Shape{3, 4}
        );

        EXPECT_EQ(shape, (ormi::core::Shape{2, 3, 4}));
    }

    TEST(ShapeTest, preservesZeroDimensionWhenBroadcastWithSingleton) {
        EXPECT_EQ(
            ormi::core::broadcast(ormi::core::Shape{0, 3}, ormi::core::Shape{1, 3}),
            (ormi::core::Shape{0, 3})
        );
        EXPECT_EQ(
            ormi::core::broadcast(ormi::core::Shape{1, 3}, ormi::core::Shape{0, 3}),
            (ormi::core::Shape{0, 3})
        );
    }

    TEST(ShapeTest, rejectsUnbroadcastableShapes) {
        EXPECT_THROW(
            (void)ormi::core::broadcast(ormi::core::Shape{2, 3}, ormi::core::Shape{4, 3}),
            ormi::ShapeError
        );
    }

    TEST(ShapeTest, streamsShape) {
        std::ostringstream out;
        out << ormi::core::Shape{2, 3};

        EXPECT_EQ(out.str(), "[2, 3]");
    }
}
