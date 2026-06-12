#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <stdexcept>

#include "tests/test_utils.h"
#include "core/autograd.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"

using namespace ormi;

class AutogradTest : public ormi::testing::HardwareTestSuite {};

TEST_F(AutogradTest, ComputeGradientsAndSeed) {
    for (auto dev : available_devices) {
        auto buf_a = full({2, 2}, 2.0f, dev);
        auto buf_b = full({2, 2}, 3.0f, dev);
        
        auto a = std::make_shared<Node>(buf_a, true, dev);
        auto b = std::make_shared<Node>(buf_b, true, dev);
        
        // Create an internal node (Z = A + B)
        auto c = std::make_shared<Node>(
            ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{a, b}, dev
        );
        
        // Gradients should be null before backward
        EXPECT_EQ(c->grad, nullptr);
        EXPECT_EQ(a->grad, nullptr);
        EXPECT_EQ(b->grad, nullptr);
        
        compute_gradients(c);
        
        // The root node (c) should be seeded with an exact shape match
        ASSERT_NE(c->grad, nullptr);
        EXPECT_EQ(c->grad->view.shape, (std::vector<int>{2, 2}));
        
        // The topological traversal should have routed through adjoint_add 
        // populated the parent gradients
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 2}));
        
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{2, 2}));
    }
}

TEST_F(AutogradTest, ZeroGradGraph) {
    auto dev = Device(DeviceEnum::CPU);
    
    auto a = std::make_shared<Node>(full({2, 2}, 2.0f, dev.type), true, dev);
    auto b = std::make_shared<Node>(full({2, 2}, 3.0f, dev.type), true, dev);
    auto c = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{a, b}, dev);
    
    // backward
    compute_gradients(c);
    ASSERT_NE(c->grad, nullptr);
    ASSERT_NE(a->grad, nullptr);
    
    zero_grad_graph(c);
    
    // grad is nullified
    EXPECT_EQ(c->grad, nullptr);
    EXPECT_EQ(a->grad, nullptr);
    EXPECT_EQ(b->grad, nullptr);
}

TEST_F(AutogradTest, MissingAdjointThrows) {
    auto dev = Device(DeviceEnum::CPU);
    
    // CONST nodes have shape inference rules but NO adjoint rules.
    // They are intentionally excluded from the backprop tape. 
    auto const_node = std::make_shared<Node>(
        ops::OpEnum::CONST, std::vector<std::shared_ptr<Node>>{}, dev
    );
    
    // compute_gradients will seed the const_node, enter the backprop loop, 
    // realize CONST is not a LEAF, and query the Registry for an adjoint.
    // When the Registry returns a nullptr, it should trigger our safety exception.
    EXPECT_THROW(compute_gradients(const_node), std::runtime_error);
}