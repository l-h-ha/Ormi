#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/buffer.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/ops/pointwise/div.h"

using namespace ormi;

class OpsDivTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsDivTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::div(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::DIV);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

        auto b_vec = make_leaf({3}, false, dev);
        auto c_broad = ops::div(a, b_vec);

        EXPECT_EQ(c_broad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsDivTest, AdjointGraphConstruction_X) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::div(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::DIV);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::DIV);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        EXPECT_EQ(a_grad->parents[0], seed);
        EXPECT_EQ(a_grad->parents[1], b);

        EXPECT_EQ(ctx.get(b), nullptr);
    }
}

TEST_F(OpsDivTest, AdjointGraphConstruction_Y) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::div(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::DIV);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto b_grad = ctx.get(b);

        ASSERT_NE(b_grad, nullptr);

        // dY = -(dZ * (X / (Y * Y)))
        EXPECT_EQ(b_grad->op, ops::OpEnum::NEG);

        auto dz_mul = b_grad->parents[0];
        ASSERT_NE(dz_mul, nullptr);

        EXPECT_EQ(dz_mul->op, ops::OpEnum::MUL);
        EXPECT_EQ(dz_mul->parents[0], seed);

        auto x_div_ysq = dz_mul->parents[1];
        ASSERT_NE(x_div_ysq, nullptr);

        EXPECT_EQ(x_div_ysq->op, ops::OpEnum::DIV);
        EXPECT_EQ(x_div_ysq->parents[0], a);

        auto y_sq = x_div_ysq->parents[1];
        ASSERT_NE(y_sq, nullptr);

        EXPECT_EQ(y_sq->op, ops::OpEnum::MUL);
        EXPECT_EQ(y_sq->parents[0], b);
        EXPECT_EQ(y_sq->parents[1], b);

        EXPECT_EQ(ctx.get(a), nullptr);
    }
}

TEST_F(OpsDivTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::div(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto existing_a_grad = make_leaf({2, 3}, false, dev);
        ctx.set(a, existing_a_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::DIV);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        EXPECT_EQ(a_grad->parents[0], existing_a_grad);
        EXPECT_EQ(a_grad->parents[1]->op, ops::OpEnum::DIV);
    }
}

TEST_F(OpsDivTest, AdjointPartialGradient) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::div(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::DIV);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        EXPECT_NE(ctx.get(a), nullptr);
        EXPECT_EQ(ctx.get(b), nullptr);
    }
}