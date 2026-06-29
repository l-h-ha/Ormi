#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/test_utils.h"

#include "core/grad_context.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/log.h"

using namespace ormi;

class OpsLogTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsLogTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto c = ops::log(a);

        EXPECT_EQ(c->op, ops::OpEnum::LOG);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsLogTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::log(a);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::LOG);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        // Since shapes match, unbroadcast returns the DIV node.
        EXPECT_EQ(a_grad->op, ops::OpEnum::DIV);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        // Verify the chain: DIV(dz, x)
        auto dz = a_grad->parents[0];
        auto x = a_grad->parents[1];

        EXPECT_EQ(dz, seed);
        EXPECT_EQ(x, a);
    }
}

TEST_F(OpsLogTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::log(a);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        // Pre-populate input grad to trigger accumulation.
        auto existing_grad = make_leaf({2, 3}, false, dev);
        ctx.set(a, existing_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::LOG);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        EXPECT_EQ(a_grad->parents[0], existing_grad);
        EXPECT_EQ(a_grad->parents[1]->op, ops::OpEnum::DIV);
    }
}

TEST_F(OpsLogTest, AdjointSkipWhenNoGradRequired) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto c = ops::log(a);

        GradContext ctx;

        auto seed = make_leaf({2, 3}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::LOG);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        EXPECT_EQ(ctx.get(a), nullptr);
    }
}