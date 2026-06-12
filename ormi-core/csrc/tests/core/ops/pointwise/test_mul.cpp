#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/mul.h"

using namespace ormi;

class OpsMulTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsMulTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::mul(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::MUL);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

        // Broadcast
        auto b_vec = make_leaf({3}, false, dev);
        auto c_broad = ops::mul(a, b_vec);
        EXPECT_EQ(c_broad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsMulTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::mul(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MUL);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        // dX = dZ * Y
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(a->grad->parents[0], c->grad);
        EXPECT_EQ(a->grad->parents[1], b);

        // dY = dZ * X
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(b->grad->parents[0], c->grad);
        EXPECT_EQ(b->grad->parents[1], a);
    }
}

TEST_F(OpsMulTest, AdjointUnbroadcasting) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3}, true, dev);
        auto c = ops::mul(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MUL);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        // Gradient for b requires unbroadcasting
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->op, ops::OpEnum::SUM);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{3}));
    }
}

TEST_F(OpsMulTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::mul(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Pre-populate input grad
        a->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MUL);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}