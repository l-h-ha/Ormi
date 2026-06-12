#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/add.h"

using namespace ormi;

class OpsAddTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsAddTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::add(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::ADD);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

        // Broadcast [2, 3] + [3] -> [2, 3]
        auto b_vec = make_leaf({3}, false, dev);
        auto c_broad = ops::add(a, b_vec);
        EXPECT_EQ(c_broad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsAddTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::add(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::ADD);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad, c->grad);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad, c->grad);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsAddTest, AdjointUnbroadcasting) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3}, true, dev);
        auto c = ops::add(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::ADD);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        // Gradient for b requires unbroadcasting
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->op, ops::OpEnum::SUM);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{3}));
    }
}

TEST_F(OpsAddTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::add(a, b);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Pre-populate input grad
        a->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::ADD);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}