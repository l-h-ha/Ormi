#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "tests/test_utils.h"

#include "core/attributes.h"
#include "core/grad_context.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/ops/view/transpose.h"
#include "core/registry.h"

using namespace ormi;

class OpsTransposeTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsTransposeTest, ZeroCopyStrideInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);

        std::vector<int> axes = {2, 0, 1};
        auto b = ops::transpose(a, axes);

        EXPECT_EQ(b->op, ops::OpEnum::TRANSPOSE);
        EXPECT_EQ(b->view.shape, (std::vector<int>{4, 2, 3}));
        EXPECT_EQ(b->view.strides, (std::vector<int>{1, 12, 4}));
    }
}

TEST_F(OpsTransposeTest, InvalidAxesThrow) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);

        std::vector<int> too_few = {1, 0};
        EXPECT_THROW(ops::transpose(a, too_few), std::runtime_error);

        std::vector<int> too_many = {0, 1, 2, 3};
        EXPECT_THROW(ops::transpose(a, too_many), std::runtime_error);
    }
}

TEST_F(OpsTransposeTest, AdjointInversePermutation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);

        std::vector<int> axes = {2, 0, 1};
        auto z = ops::transpose(a, axes);

        GradContext ctx;

        auto seed = make_leaf({4, 2, 3}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::TRANSPOSE);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::TRANSPOSE);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3, 4}));

        const auto& inv_attrs = std::get<PermuteAttrs>(a_grad->attrs);
        EXPECT_EQ(inv_attrs.axes, (std::vector<int>{1, 2, 0}));
    }
}

TEST_F(OpsTransposeTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto z = ops::transpose(a, {1, 0});

        GradContext ctx;

        auto seed = make_leaf({3, 2}, false, dev);
        ctx.set(z, seed);

        auto existing_grad = make_leaf({2, 3}, false, dev);
        ctx.set(a, existing_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::TRANSPOSE);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsTransposeTest, AdjointSkipWhenNoGradRequired) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto z = ops::transpose(a, {1, 0});

        GradContext ctx;

        auto seed = make_leaf({3, 2}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::TRANSPOSE);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        EXPECT_EQ(ctx.get(a), nullptr);

    }
}