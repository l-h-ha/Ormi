#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <algorithm>

#include "core/node.h"
#include "core/ops.h"
#include "core/buffer.h"
#include "core/dtype.h"
#include "core/tapes/tape.h"

using namespace ormi;

class TopologySortTest : public ::testing::Test {
protected:
    std::shared_ptr<Node> make_leaf() {
        auto dev = Device(DeviceEnum::CPU);
        auto buf = empty({1}, dev, DType::Float32);
        return std::make_shared<Node>(buf, false);
    }

    std::shared_ptr<Node> make_internal(ops::OpEnum op, const std::vector<std::shared_ptr<Node>>& parents) {
        return std::make_shared<Node>(op, parents);
    }

    bool appears_before(const std::vector<std::shared_ptr<Node>>& sorted, 
                        const std::shared_ptr<Node>& first, 
                        const std::shared_ptr<Node>& second) {
        auto it_first = std::find(sorted.begin(), sorted.end(), first);
        auto it_second = std::find(sorted.begin(), sorted.end(), second);
        
        if (it_first == sorted.end() || it_second == sorted.end()) {
            return false;
        }
        return it_first < it_second;
    }
};

TEST_F(TopologySortTest, SingleNode) {
    auto a = make_leaf();
    
    auto sorted = Topology::sort({a});
    
    ASSERT_EQ(sorted.size(), 1);
    EXPECT_EQ(sorted[0], a);
}

TEST_F(TopologySortTest, LinearChain) {
    // Graph: A -> B -> C
    // C depends on B, B depends on A
    auto a = make_leaf();
    auto b = make_internal(ops::OpEnum::COS, {a});
    auto c = make_internal(ops::OpEnum::SIN, {b});

    auto sorted = Topology::sort({c});

    ASSERT_EQ(sorted.size(), 3);
    
    // parents must precede children
    EXPECT_EQ(sorted[0], a);
    EXPECT_EQ(sorted[1], b);
    EXPECT_EQ(sorted[2], c);
}

TEST_F(TopologySortTest, DiamondDependency) {
    // Shared dependency
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D
    auto a = make_leaf();
    auto b = make_internal(ops::OpEnum::COS, {a});
    auto c = make_internal(ops::OpEnum::SIN, {a});
    auto d = make_internal(ops::OpEnum::ADD, {b, c});

    auto sorted = Topology::sort({d});

    // Ensure uniqueness
    ASSERT_EQ(sorted.size(), 4);
    
    // structural validation
    EXPECT_TRUE(appears_before(sorted, a, b));
    EXPECT_TRUE(appears_before(sorted, a, c));
    EXPECT_TRUE(appears_before(sorted, b, d));
    EXPECT_TRUE(appears_before(sorted, c, d));
    
    // D must always be the final element evaluated
    EXPECT_EQ(sorted.back(), d);
}

TEST_F(TopologySortTest, MultipleDisconnectedRoots) {
    // Graph 1: A -> B
    auto a = make_leaf();
    auto b = make_internal(ops::OpEnum::COS, {a});

    // Graph 2: C -> D
    auto c = make_leaf();
    auto d = make_internal(ops::OpEnum::SIN, {c});

    // Pass multiple target nodes to the sorting system
    auto sorted = Topology::sort({b, d});

    ASSERT_EQ(sorted.size(), 4);
    
    // Verify relative dependencies hold true within their respective sub-graphs
    EXPECT_TRUE(appears_before(sorted, a, b));
    EXPECT_TRUE(appears_before(sorted, c, d));
}

TEST_F(TopologySortTest, EmptyAndNullBorders) {
    auto empty_sort = Topology::sort({});
    EXPECT_TRUE(empty_sort.empty());

    // Ensure it filters out or safely ignores the nullptr 
    std::vector<std::shared_ptr<Node>> mixed_roots = { nullptr, make_leaf() };
    auto mixed_sort = Topology::sort(mixed_roots);
    ASSERT_EQ(mixed_sort.size(), 1);
    EXPECT_NE(mixed_sort[0], nullptr);
}