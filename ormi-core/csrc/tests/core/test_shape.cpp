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

    TEST(ShapeTest, zeroDimensionMakesShapeEmpty) {
        const ormi::core::Shape shape{2, 0, 4};

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
