#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "core/shape.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/attributes.h"
#include "core/error.h"
#include "core/buffer.h"
#include "core/allocator.h"
#include "core/dtype.h"

using namespace ormi;

class ShapeTest : public ::testing::Test {
protected:
    std::shared_ptr<Node> make_shape_node(const std::vector<int>& shape) {
        size_t elements = 1;
        for (int d : shape) elements *= d;
        size_t bytes = elements * sizeof(float);

        auto* allocator = get_allocator(Device::cpu());
        void* ptr = allocator->allocate(bytes);

        auto dummy_mem = std::make_shared<Buffer>(
            ptr, bytes, shape, Device::cpu(), DType::Float32
        );

        return std::make_shared<Node>(dummy_mem, false);
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
    EXPECT_THROW(broadcast_shapes({2}, {3}), ormi::ShapeError);

    // Mismatched inner dimensions
    EXPECT_THROW(broadcast_shapes({2, 3}, {2, 4}), ormi::ShapeError);

    // Transposed dimensions cannot broadcast automatically
    EXPECT_THROW(broadcast_shapes({3, 2}, {2, 3}), ormi::ShapeError);
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
    auto p1 = make_shape_node({2, 2});

    ops::OpEnum invalid_op = static_cast<ops::OpEnum>(9999);
    EXPECT_THROW(infer_shape(invalid_op, {p1}, empty_attrs), ormi::RegistryError);
}

TEST_F(ShapeTest, InternalNodeEmptyParentsThrows) {
    OpAttributes empty_attrs;
    std::vector<std::shared_ptr<Node>> empty_parents;

    EXPECT_THROW(
        (void)std::make_shared<Node>(ops::OpEnum::ADD, empty_parents, empty_attrs),
        ormi::NodeError
    );
}

TEST_F(ShapeTest, TensorViewContiguity) {
    // [2, 3] with strides [3, 1]
    // 3*2 = 6 total elements.
    // Strides: index 1 is 1 (matches 1), index 0 is 3 (matches shape[1]*stride[1] = 3*1=3)
    auto contiguous_node = make_shape_node({2, 3});
    contiguous_node->view.strides = {3, 1};
    EXPECT_TRUE(contiguous_node->view.is_contiguous());

    // [2, 3] with swapped strides [1, 3]
    auto non_contiguous_node = make_shape_node({2, 3});
    non_contiguous_node->view.strides = {1, 3};
    EXPECT_FALSE(non_contiguous_node->view.is_contiguous());

    // Empty
    auto scalar_node = make_shape_node({});
    EXPECT_TRUE(scalar_node->view.is_contiguous());
}