#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "backend/cpu/jit.h"
#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/pointwise/sin.h"

using namespace ormi;

class CPUJitTest : public ::testing::Test {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, float val) {
        auto dev = Device(DeviceEnum::CPU);
        auto buf = full(shape, val, dev.type);
        return std::make_shared<Node>(buf, false, dev);
    }
};

TEST_F(CPUJitTest, BasicExecution) {
    auto a = make_leaf({4}, 2.0f);
    auto b = make_leaf({4}, 3.0f);
    auto c = ops::add(a, b);

    cpu::execute_jit(c);

    EXPECT_TRUE(c->is_realized());
    EXPECT_EQ(c->view.shape, (std::vector<int>{4}));

    auto* out = static_cast<float*>(c->data->ptr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], 5.0f);
    }
}

TEST_F(CPUJitTest, TapeChaining) {
    auto a = make_leaf({2, 2}, 2.0f);
    auto b = make_leaf({2, 2}, 3.0f);
    auto c = make_leaf({2, 2}, 4.0f);

    // D = (A + B) * C
    auto a_plus_b = ops::add(a, b);
    auto d = ops::mul(a_plus_b, c);

    cpu::execute_jit(d);

    auto* out = static_cast<float*>(d->data->ptr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], 20.0f);
    }
}

TEST_F(CPUJitTest, BroadcastingExecution) {
    // A: [2, 3] + B: [3] -> [2, 3]
    auto a = make_leaf({2, 3}, 2.0f);
    auto b = make_leaf({3}, 5.0f);
    auto c = ops::add(a, b);

    cpu::execute_jit(c);

    EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

    auto* out = static_cast<float*>(c->data->ptr);
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out[i], 7.0f);
    }
}

TEST_F(CPUJitTest, ScalarExecution) {
    // A: [2, 2] + B: [] -> [2, 2]
    auto a = make_leaf({2, 2}, 10.0f);
    auto b = make_leaf({}, 3.0f); 
    auto c = ops::add(a, b);

    cpu::execute_jit(c);

    auto* out = static_cast<float*>(c->data->ptr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], 13.0f);
    }
}

TEST_F(CPUJitTest, UnaryMathFunctions) {
    // PI / 2 (approx)
    auto a = make_leaf({2}, 1.570796327f); 
    auto b = ops::sin(a);

    cpu::execute_jit(b);

    auto* out = static_cast<float*>(b->data->ptr);
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(out[i], 1.0f, 1e-5f); 
    }
}