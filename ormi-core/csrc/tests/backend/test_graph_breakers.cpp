#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "backend/graph_breakers.h"

using namespace ormi;

class GraphBreakerTest : public ::testing::Test {
protected:
    std::shared_ptr<Node> make_dummy_node(ops::OpEnum op, DeviceEnum dev_type) {
        auto dev = Device(dev_type);
        auto buf = empty({2, 2}, dev_type);
        auto node = std::make_shared<Node>(buf, false, dev);
        // Manually override the OP enum for testing interception without triggering shape rules
        node->op = op; 
        return node;
    }
};

TEST_F(GraphBreakerTest, SuccessfulInterception) {
    ops::OpEnum dummy_op = static_cast<ops::OpEnum>(9999);
    bool handler_called = false;
    
    // Register custom breaker
    RegisterGraphBreaker reg(DeviceEnum::CPU, dummy_op, [&](std::shared_ptr<Node> n) {
        handler_called = true;
    });

    auto node = make_dummy_node(dummy_op, DeviceEnum::CPU);
    bool intercepted = try_execute_graph_break(node);
    
    EXPECT_TRUE(intercepted);
    EXPECT_TRUE(handler_called);
}

TEST_F(GraphBreakerTest, DeviceIsolation) {
    ops::OpEnum dummy_op = static_cast<ops::OpEnum>(8888);
    
    // Register ONLY for CUDA
    RegisterGraphBreaker reg(DeviceEnum::CUDA, dummy_op, [](std::shared_ptr<Node>) {});

    // Try executing on CPU
    auto node_cpu = make_dummy_node(dummy_op, DeviceEnum::CPU);
    bool intercepted_cpu = try_execute_graph_break(node_cpu);
    
    // Should fall through to JIT because the device key mismatches
    EXPECT_FALSE(intercepted_cpu);
}

TEST_F(GraphBreakerTest, FallthroughUnregistered) {
    // Basic element-wise operations do not register graph breakers
    auto node = make_dummy_node(ops::OpEnum::ADD, DeviceEnum::CPU);
    
    bool intercepted = try_execute_graph_break(node);
    EXPECT_FALSE(intercepted);
}