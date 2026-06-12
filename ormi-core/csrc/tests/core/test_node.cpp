#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/attributes.h"

using namespace ormi;

class NodeTest : public ormi::testing::HardwareTestSuite {};

TEST_F(NodeTest, LeafNodeInitialization) {
    for (auto dev : available_devices) {
        auto buf = empty({2, 3, 4}, dev);
        auto node = std::make_shared<Node>(buf, true, dev);

        EXPECT_EQ(node->op, ops::OpEnum::LEAF);
        EXPECT_TRUE(node->requires_grad);
        EXPECT_TRUE(node->is_realized());
        EXPECT_NE(node->data, nullptr);

        // View should match the physical buffer
        EXPECT_EQ(node->view.shape, (std::vector<int>{2, 3, 4}));
        EXPECT_EQ(node->view.strides, (std::vector<int>{12, 4, 1}));
    }
}

TEST_F(NodeTest, RequiresGradInheritance) {
    auto dev = Device(DeviceEnum::CPU); 
    
    auto buf1 = empty({2}, dev.type);
    auto buf2 = empty({2}, dev.type);

    auto leaf_false = std::make_shared<Node>(buf1, false, dev);
    auto leaf_true = std::make_shared<Node>(buf2, true, dev);

    // False + False = False
    auto node_ff = std::make_shared<Node>(
        ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{leaf_false, leaf_false}, dev
    );
    EXPECT_FALSE(node_ff->requires_grad);

    // True + False = True
    auto node_tf = std::make_shared<Node>(
        ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{leaf_true, leaf_false}, dev
    );
    EXPECT_TRUE(node_tf->requires_grad);

    // True + True = True
    auto node_tt = std::make_shared<Node>(
        ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{leaf_true, leaf_true}, dev
    );
    EXPECT_TRUE(node_tt->requires_grad);
}

TEST_F(NodeTest, DefaultStrideInference) {
    auto dev = Device(DeviceEnum::CPU);
    auto buf = empty({4, 5}, dev.type);
    auto a = std::make_shared<Node>(buf, false, dev);
    auto b = std::make_shared<Node>(buf, false, dev);

    // ADD has no RegisterStride hook, so it must default to contiguous
    auto c = std::make_shared<Node>(
        ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{a, b}, dev
    );

    EXPECT_EQ(c->view.shape, (std::vector<int>{4, 5}));
    EXPECT_EQ(c->view.strides, (std::vector<int>{5, 1}));
}

TEST_F(NodeTest, CustomStrideInference) {
    auto dev = Device(DeviceEnum::CPU);
    // Base Shape [2, 3, 4], Contiguous Strides [12, 4, 1]
    auto buf = empty({2, 3, 4}, dev.type);
    auto a = std::make_shared<Node>(buf, false, dev);

    // Transpose axes (0, 2, 1) -> new shape should be [2, 4, 3]
    PermuteAttrs attrs{std::vector<int>{0, 2, 1}};
    
    // TRANSPOSE has a custom RegisterStride hook. 
    auto transposed = std::make_shared<Node>(
        ops::OpEnum::TRANSPOSE, std::vector<std::shared_ptr<Node>>{a}, dev, attrs
    );

    // Expect strides to be permuted the same way as the shape: [12, 1, 4]
    EXPECT_EQ(transposed->view.shape, (std::vector<int>{2, 4, 3}));
    EXPECT_EQ(transposed->view.strides, (std::vector<int>{12, 1, 4}));
}

TEST_F(NodeTest, RealizationState) {
    for (auto dev : available_devices) {
        auto a = std::make_shared<Node>(empty({2, 2}, dev), false, dev);
        auto b = std::make_shared<Node>(empty({2, 2}, dev), false, dev);

        auto c = std::make_shared<Node>(
            ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{a, b}, dev
        );

        // memory is not allocated yet
        EXPECT_FALSE(c->is_realized());
        EXPECT_EQ(c->data, nullptr);

        // JIT
        c->realize();

        // Memory is allocated and graph is executed
        EXPECT_TRUE(c->is_realized());
        EXPECT_NE(c->data, nullptr);
    }
}