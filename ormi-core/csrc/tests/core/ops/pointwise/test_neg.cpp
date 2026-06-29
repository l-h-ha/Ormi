#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/buffer.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/ops/pointwise/neg.h"

using namespace ormi;

class OpsNegTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsNegTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto c = ops::neg(a);

        EXPECT_EQ(c->op, ops::OpEnum::NEG);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsNegTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::neg(a);

        GradContext ctx;

        // Seed gradient
        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::NEG);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        // dX = -dZ
        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::NEG);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(a_grad->parents[0], seed);
    }
}

TEST_F(OpsNegTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::neg(a);

        GradContext ctx;

        // Seed output gradient
        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        // Pre-populate input grad inside GradContext
        auto existing_grad = make_leaf({2, 3}, false, dev);
        ctx.set(a, existing_grad);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::NEG);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(a_grad->parents[0], existing_grad);
        EXPECT_EQ(a_grad->parents[1]->op, ops::OpEnum::NEG);
    }
}

TEST_F(OpsNegTest, AdjointSkipWhenNoGradRequired) {
    for (auto dev : available_devices) {
        // A does NOT require grad
        auto a = make_leaf({2, 3}, false, dev);
        auto c = ops::neg(a);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::NEG);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        // A should remain without a gradient
        EXPECT_EQ(ctx.get(a), nullptr);
    }
}