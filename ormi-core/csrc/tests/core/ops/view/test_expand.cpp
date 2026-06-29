#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/test_utils.h"

#include "core/attributes.h"
#include "core/dtype.h"
#include "core/grad_context.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/ops/view/expand.h"
#include "core/registry.h"

using namespace ormi;

class OpsExpandTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsExpandTest, ZeroCopyStrideInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({3, 4}, false, dev);
        auto b = ops::expand(a, {2, 3, 4});

        EXPECT_EQ(b->op, ops::OpEnum::EXPAND);
        EXPECT_EQ(b->view.shape, (std::vector<int>{2, 3, 4}));
        EXPECT_EQ(b->view.strides, (std::vector<int>{0, 4, 1}));

        auto c = make_leaf({2, 1, 4}, false, dev);
        auto d = ops::expand(c, {2, 5, 4});

        EXPECT_EQ(d->view.shape, (std::vector<int>{2, 5, 4}));
        EXPECT_EQ(d->view.strides, (std::vector<int>{4, 0, 1}));
    }
}

TEST_F(OpsExpandTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({1, 4}, true, dev);
        auto z = ops::expand(a, {3, 4});

        GradContext ctx;

        auto seed = make_leaf({3, 4}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::EXPAND);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::SUM);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{1, 4}));

        const auto& attrs = std::get<ReduceAttrs>(a_grad->attrs);
        EXPECT_EQ(attrs.axes, (std::vector<int>{0}));
    }
}

TEST_F(OpsExpandTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({1, 4}, true, dev);
        auto z = ops::expand(a, {3, 4});

        GradContext ctx;

        auto seed = make_leaf({3, 4}, false, dev);
        auto existing_grad = make_leaf({1, 4}, false, dev);

        ctx.set(z, seed);
        ctx.set(a, existing_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::EXPAND);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{1, 4}));
    }
}

TEST_F(OpsExpandTest, AdjointSkipWhenNoGradRequired) {
    for (auto dev : available_devices) {
        auto a = make_leaf({1, 4}, false, dev);
        auto z = ops::expand(a, {3, 4});

        GradContext ctx;

        auto seed = make_leaf({3, 4}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::EXPAND);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        EXPECT_EQ(ctx.get(a), nullptr);
    }
}