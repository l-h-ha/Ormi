#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

#include "core/shape.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/attributes.h"

using namespace ormi;

class ShapeTest : public ::testing::Test {
protected:
    std::shared_ptr<Node> make_shape_node(const std::vector<int>& shape) {
        auto node = std::make_shared<Node>(
            ops::OpEnum::LEAF, 
            std::vector<std::shared_ptr<Node>>{}, 
            Device(DeviceEnum::CPU)
        );
        node->view.shape = shape;
        return node;
    }
};

TEST_F(ShapeTest, CalcContiguousStrides) {
    // Tensor: [2, 3, 4] -> Strides should be [12, 4, 1]
    EXPECT_EQ(calc_contiguous_strides({2, 3, 4}), (std::vector<int>{12, 4, 1}));

    // Matrix: [5, 5] -> Strides should be [5, 1]
    EXPECT_EQ(calc_contiguous_strides({5, 5}), (std::vector<int>{5, 1}));

    // Vector: [10] -> Strides should be [1]
    EXPECT_EQ(calc_contiguous_strides({10}), (std::vector<int>{1}));

    // Scalar: [] -> Strides should be []
    EXPECT_EQ(calc_contiguous_strides({}), (std::vector<int>{}));
}

TEST_F(ShapeTest, BroadcastShapesSuccess) {
    // Identical shapes
    EXPECT_EQ(broadcast_shapes({2, 3}, {2, 3}), (std::vector<int>{2, 3}));

    // Scalar broadcast
    EXPECT_EQ(broadcast_shapes({2, 3}, {}), (std::vector<int>{2, 3}));
    EXPECT_EQ(broadcast_shapes({}, {4, 5, 6}), (std::vector<int>{4, 5, 6}));

    // 1D to 2D broadcast
    // [2, 3] + [3] -> [2, 3]
    EXPECT_EQ(broadcast_shapes({2, 3}, {3}), (std::vector<int>{2, 3}));

    // Missing leading dimensions
    // [2, 1, 4] + [3, 4] -> [2, 3, 4]
    EXPECT_EQ(broadcast_shapes({2, 1, 4}, {3, 4}), (std::vector<int>{2, 3, 4}));

    // inner broadcast
    // [1, 5, 1] + [4, 1, 6] -> [4, 5, 6]
    EXPECT_EQ(broadcast_shapes({1, 5, 1}, {4, 1, 6}), (std::vector<int>{4, 5, 6}));
}

TEST_F(ShapeTest, BroadcastShapesFailure) {
    // Completely mismatched dimensions
    EXPECT_THROW(broadcast_shapes({2}, {3}), std::runtime_error);

    // Mismatched inner dimensions
    EXPECT_THROW(broadcast_shapes({2, 3}, {2, 4}), std::runtime_error);

    // Transposed dimensions cannot broadcast automatically
    EXPECT_THROW(broadcast_shapes({3, 2}, {2, 3}), std::runtime_error);
}

TEST_F(ShapeTest, InferUnaryShape) {
    auto parent = make_shape_node({4, 4});
    OpAttributes empty_attrs;
    
    auto out_shape = infer_unary_shape({parent}, empty_attrs);
    EXPECT_EQ(out_shape, (std::vector<int>{4, 4}));
}

TEST_F(ShapeTest, InferBinaryShape) {
    auto p1 = make_shape_node({2, 3});
    auto p2 = make_shape_node({3});
    OpAttributes empty_attrs;
    
    // Should hit the broadcast_shapes logic internally
    auto out_shape = infer_binary_shape({p1, p2}, empty_attrs);
    EXPECT_EQ(out_shape, (std::vector<int>{2, 3}));
}

TEST_F(ShapeTest, InferShapeRegistry) {
    OpAttributes empty_attrs;
    std::vector<std::shared_ptr<Node>> empty_parents;

    // LEAF/CONST with empty parents safely returns an empty shape
    EXPECT_EQ(infer_shape(ops::OpEnum::LEAF, empty_parents, empty_attrs), (std::vector<int>{}));

    // Force an unregistered op
    ops::OpEnum invalid_op = static_cast<ops::OpEnum>(9999);
    auto p1 = make_shape_node({2, 2});
    EXPECT_THROW(infer_shape(invalid_op, {p1}, empty_attrs), std::runtime_error);
}