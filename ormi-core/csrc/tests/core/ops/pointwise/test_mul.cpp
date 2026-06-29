#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/buffer.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/ops/pointwise/mul.h"

using namespace ormi;

class OpsMulTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsMulTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::mul(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::MUL);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

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

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MUL);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(a_grad->parents[0], seed);
        EXPECT_EQ(a_grad->parents[1], b);

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(b_grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(b_grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(b_grad->parents[0], seed);
        EXPECT_EQ(b_grad->parents[1], a);
    }
}

TEST_F(OpsMulTest, AdjointUnbroadcasting) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3}, true, dev);
        auto c = ops::mul(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MUL);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(b_grad->op, ops::OpEnum::SUM);
        EXPECT_EQ(b_grad->view.shape, (std::vector<int>{3}));
    }
}

TEST_F(OpsMulTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::mul(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto existing_grad = make_leaf({2, 3}, false, dev);
        ctx.set(a, existing_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MUL);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(a_grad->parents[0], existing_grad);
    }
}

TEST_F(OpsMulTest, AdjointPartialGradient) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::mul(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MUL);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        EXPECT_NE(a_grad, nullptr);
        EXPECT_EQ(b_grad, nullptr);
    }
}