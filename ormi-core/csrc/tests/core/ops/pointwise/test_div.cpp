#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/div.h"

using namespace ormi;

class OpsDivTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsDivTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::div(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::DIV);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

        // Broadcast [2, 3] / [3] -> [2, 3]
        auto b_vec = make_leaf({3}, false, dev);
        auto c_broad = ops::div(a, b_vec);
        EXPECT_EQ(c_broad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsDivTest, AdjointGraphConstruction_X) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, false, dev); // Only test dX
        auto c = ops::div(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::DIV);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        // dX = dZ / Y
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::DIV);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(a->grad->parents[0], c->grad); // dZ
        EXPECT_EQ(a->grad->parents[1], b);       // Y
    }
}

TEST_F(OpsDivTest, AdjointGraphConstruction_Y) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev); // Only test dY
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::div(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::DIV);
        adjoint_func(c);

        // dY = -dZ * (X / (Y * Y))
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->op, ops::OpEnum::NEG);
        
        auto dz_mul = b->grad->parents[0];
        EXPECT_EQ(dz_mul->op, ops::OpEnum::MUL);
        EXPECT_EQ(dz_mul->parents[0], c->grad); // dZ
        
        auto x_div_ysq = dz_mul->parents[1];
        EXPECT_EQ(x_div_ysq->op, ops::OpEnum::DIV);
        EXPECT_EQ(x_div_ysq->parents[0], a); // X
        
        auto y_sq = x_div_ysq->parents[1];
        EXPECT_EQ(y_sq->op, ops::OpEnum::MUL);
        EXPECT_EQ(y_sq->parents[0], b); // Y
        EXPECT_EQ(y_sq->parents[1], b); // Y
    }
}

TEST_F(OpsDivTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::div(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Pre-populate input grad
        a->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::DIV);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}