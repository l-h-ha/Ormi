#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/test_utils.h"

#include "core/grad_context.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/sub.h"

using namespace ormi;

class OpsSubTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(
        const std::vector<int>& shape,
        bool requires_grad,
        Device dev
    ) {
        auto buf = empty(shape, dev, DType::Float32);
        return std::make_shared<Node>(buf, requires_grad);
    }
};

TEST_F(OpsSubTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::sub(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::SUB);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

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

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUB);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad, seed);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(b_grad->op, ops::OpEnum::NEG);
        EXPECT_EQ(b_grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(b_grad->parents[0], seed);
    }
}

TEST_F(OpsSubTest, AdjointUnbroadcasting) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3}, true, dev);
        auto c = ops::sub(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUB);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(b_grad->op, ops::OpEnum::SUM);
        EXPECT_EQ(b_grad->view.shape, (std::vector<int>{3}));
        EXPECT_EQ(b_grad->parents[0]->op, ops::OpEnum::NEG);
    }
}

TEST_F(OpsSubTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::sub(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        auto existing_a_grad = make_leaf({2, 3}, false, dev);
        auto existing_b_grad = make_leaf({2, 3}, false, dev);

        ctx.set(c, seed);
        ctx.set(a, existing_a_grad);
        ctx.set(b, existing_b_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUB);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(b_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(b_grad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsSubTest, AdjointPartialGradient) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::sub(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUB);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        EXPECT_NE(ctx.get(a), nullptr);
        EXPECT_EQ(ctx.get(b), nullptr);
    }
}