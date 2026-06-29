#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/buffer.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/grad_context.h"
#include "core/ops/pointwise/cos.h"

using namespace ormi;

class OpsCosTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsCosTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto c = ops::cos(a);

        EXPECT_EQ(c->op, ops::OpEnum::COS);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsCosTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::cos(a);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::COS);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        // Unbroadcast skips because shapes match, so local_dx is exactly the MUL node.
        EXPECT_EQ(a_grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        // Verify the chain: MUL(NEG(SIN(x)), dz)
        auto neg_sin_x = a_grad->parents[0];
        auto dz = a_grad->parents[1];

        EXPECT_EQ(dz, seed);
        EXPECT_EQ(neg_sin_x->op, ops::OpEnum::NEG);

        auto sin_x = neg_sin_x->parents[0];
        EXPECT_EQ(sin_x->op, ops::OpEnum::SIN);
        EXPECT_EQ(sin_x->parents[0], a);
    }
}

TEST_F(OpsCosTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::cos(a);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        auto existing_grad = make_leaf({2, 3}, false, dev);

        ctx.set(c, seed);
        ctx.set(a, existing_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::COS);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        EXPECT_EQ(a_grad->parents[0], existing_grad);
        EXPECT_EQ(a_grad->parents[1]->op, ops::OpEnum::MUL);
    }
}

TEST_F(OpsCosTest, AdjointSkipWhenNoGradRequired) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto c = ops::cos(a);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::COS);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        EXPECT_EQ(ctx.get(a), nullptr);
    }
}