#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/sub.h"

using namespace ormi;

class OpsSubTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsSubTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::sub(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::SUB);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

        // Broadcast
        auto b_vec = make_leaf({3}, false, dev);
        auto c_broad = ops::sub(a, b_vec);
        EXPECT_EQ(c_broad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsSubTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::sub(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUB);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        // dX = dZ
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad, c->grad);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        // dY = -dZ
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->op, ops::OpEnum::NEG);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(b->grad->parents[0], c->grad);
    }
}

TEST_F(OpsSubTest, AdjointUnbroadcasting) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3}, true, dev);
        auto c = ops::sub(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUB);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        // Gradient for b requires unbroadcasting after negation
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->op, ops::OpEnum::SUM);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{3}));
        EXPECT_EQ(b->grad->parents[0]->op, ops::OpEnum::NEG);
    }
}

TEST_F(OpsSubTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::sub(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Pre-populate input grads
        a->grad = make_leaf({2, 3}, false, dev);
        b->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUB);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{2, 3}));
    }
}