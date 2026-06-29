#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/test_utils.h"

#include "core/grad_context.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/pow.h"

using namespace ormi;

class OpsPowTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsPowTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::pow(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::POW);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsPowTest, AdjointGraphConstruction_BaseGrad) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::pow(a, b);

        GradContext ctx;
        ctx.set(c, make_leaf({2, 3}, false, dev));

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::POW);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(b_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsPowTest, AdjointGraphConstruction_ExpGrad) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::pow(a, b);

        GradContext ctx;
        ctx.set(c, make_leaf({2, 3}, false, dev));

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::POW);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(a_grad, nullptr);

        EXPECT_EQ(b_grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(b_grad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsPowTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::pow(a, b);

        GradContext ctx;
        ctx.set(c, make_leaf({2, 3}, false, dev));

        ctx.set(a, make_leaf({2, 3}, false, dev));
        ctx.set(b, make_leaf({2, 3}, false, dev));

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::POW);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(b_grad->op, ops::OpEnum::ADD);
    }
}

TEST_F(OpsPowTest, AdjointPartialGradient) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::pow(a, b);

        GradContext ctx;
        ctx.set(c, make_leaf({2, 3}, false, dev));

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::POW);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        EXPECT_NE(a_grad, nullptr);
        EXPECT_EQ(b_grad, nullptr);
    }
}