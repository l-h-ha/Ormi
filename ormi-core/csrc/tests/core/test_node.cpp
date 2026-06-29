#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/buffer.h"
#include "core/ops.h"
#include "core/attributes.h"

using namespace ormi;

class NodeTest : public ormi::testing::HardwareTestSuite {};

TEST_F(NodeTest, LeafNodeInitialization) {
    for (auto dev : available_devices) {
        auto buf = empty({2, 3, 4}, dev, DType::Float32);
        auto node = std::make_shared<Node>(buf, true);

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
    auto dev = Device::cpu();

    auto buf1 = empty({2}, dev, DType::Float32);
    auto buf2 = empty({2}, dev, DType::Float32);

    auto leaf_false = std::make_shared<Node>(buf1, false);
    auto leaf_true = std::make_shared<Node>(buf2, true);

    // False + False = False
    auto node_ff = std::make_shared<Node>(
        ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{leaf_false, leaf_false}
    );
    EXPECT_FALSE(node_ff->requires_grad);

    // True + False = True
    auto node_tf = std::make_shared<Node>(
        ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{leaf_true, leaf_false}
    );
    EXPECT_TRUE(node_tf->requires_grad);

    // True + True = True
    auto node_tt = std::make_shared<Node>(
        ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{leaf_true, leaf_true}
    );
    EXPECT_TRUE(node_tt->requires_grad);
}

TEST_F(NodeTest, DefaultStrideInference) {
    auto dev = Device::cpu();
    auto buf = empty({4, 5}, dev, DType::Float32);
    auto a = std::make_shared<Node>(buf, false);
    auto b = std::make_shared<Node>(buf, false);

    // ADD has no RegisterStride hook, so it must default to contiguous
    auto c = std::make_shared<Node>(
        ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{a, b}
    );

    EXPECT_EQ(c->view.shape, (std::vector<int>{4, 5}));
    EXPECT_EQ(c->view.strides, (std::vector<int>{5, 1}));
}

TEST_F(NodeTest, CustomStrideInference) {
    auto dev = Device::cpu();
    // Base Shape [2, 3, 4], Contiguous Strides [12, 4, 1]
    auto buf = empty({2, 3, 4}, dev, DType::Float32);
    auto a = std::make_shared<Node>(buf, false);

    // Transpose axes (0, 2, 1) -> new shape should be [2, 4, 3]
    PermuteAttrs attrs{std::vector<int>{0, 2, 1}};

    // TRANSPOSE has a custom RegisterStride hook.
    auto transposed = std::make_shared<Node>(
        ops::OpEnum::TRANSPOSE, std::vector<std::shared_ptr<Node>>{a}, attrs
    );

    // Expect strides to be permuted the same way as the shape: [12, 1, 4]
    EXPECT_EQ(transposed->view.shape, (std::vector<int>{2, 4, 3}));
    EXPECT_EQ(transposed->view.strides, (std::vector<int>{12, 1, 4}));
}

TEST_F(NodeTest, RealizationState) {
    for (auto dev : available_devices) {
        auto a = std::make_shared<Node>(empty({2, 2}, dev, DType::Float32), false);
        auto b = std::make_shared<Node>(empty({2, 2}, dev, DType::Float32), false);

        auto c = std::make_shared<Node>(
            ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{a, b}
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

TEST_F(NodeTest, ConstructorValidationThrows) {
    auto dev = Device::cpu();
    auto node_a = std::make_shared<Node>(empty({2}, dev, DType::Float32), false);

    // DType Mismatch
    auto node_int = std::make_shared<Node>(empty({2}, dev, DType::Int32), false);
    EXPECT_THROW(
        (void)std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{node_a, node_int}),
        ormi::DTypeError
    );

    // Empty Parents
    EXPECT_THROW(
        (void)std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{}),
        ormi::NodeError
    );
}

TEST_F(NodeTest, DeviceMismatch) {
    bool has_cuda = false;
        for (auto dev : available_devices) {
            if (dev.type == DeviceEnum::CUDA) has_cuda = true;
        }
        if (!has_cuda) GTEST_SKIP() << "Skipping node device mismatch test: CUDA device unavailable.";

    auto dev = Device::cpu();
    auto node_a = std::make_shared<Node>(empty({2}, dev, DType::Float32), false);
    auto node_gpu = std::make_shared<Node>(empty({2}, Device::cuda(), DType::Float32), false);
    EXPECT_THROW(
        (void)std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{node_a, node_gpu}),
        ormi::DeviceError
    );
}

TEST_F(NodeTest, NodeNaming) {
    auto dev = Device::cpu();
    auto node = std::make_shared<Node>(empty({2}, dev, DType::Float32), false, "A");

    EXPECT_EQ(node->name, "A");

    node->set_name("B");
    EXPECT_EQ(node->name, "B");
}