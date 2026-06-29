#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <algorithm>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/buffer.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/attributes.h"
#include "core/ops/reduce/sum.h"
#include "core/grad_context.h"
#include "core/dtype.h"

using namespace ormi;

class OpsSumTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsSumTest, GlobalReductionShape) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);

        auto clean_sum = ops::sum(a, {}, false);
        EXPECT_EQ(clean_sum->op, ops::OpEnum::SUM);
        EXPECT_TRUE(clean_sum->view.shape.empty());

        auto keep_sum = ops::sum(a, {}, true);
        EXPECT_EQ(keep_sum->view.shape, (std::vector<int>{1, 1, 1}));
    }
}

TEST_F(OpsSumTest, AxisSpecificReductionShape) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);

        auto sum_ax1 = ops::sum(a, {1}, false);
        EXPECT_EQ(sum_ax1->view.shape, (std::vector<int>{2, 4}));

        auto sum_ax1_keep = ops::sum(a, {1}, true);
        EXPECT_EQ(sum_ax1_keep->view.shape, (std::vector<int>{2, 1, 4}));

        auto sum_ax02 = ops::sum(a, {0, 2}, false);
        EXPECT_EQ(sum_ax02->view.shape, (std::vector<int>{3}));

        auto sum_ax02_keep = ops::sum(a, {0, 2}, true);
        EXPECT_EQ(sum_ax02_keep->view.shape, (std::vector<int>{1, 3, 1}));
    }
}

TEST_F(OpsSumTest, AdjointWithKeepDims) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);

        auto z = ops::sum(a, {1}, true);

        GradContext ctx;

        auto seed = make_leaf({2, 1, 4}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUM);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::EXPAND);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3, 4}));

        const auto& exp_attrs = std::get<ExpandAttrs>(a_grad->attrs);
        EXPECT_EQ(exp_attrs.shape, (std::vector<int>{2, 3, 4}));
    }
}

TEST_F(OpsSumTest, AdjointWithoutKeepDims) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);

        auto z = ops::sum(a, {1}, false);

        GradContext ctx;

        auto seed = make_leaf({2, 4}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUM);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::EXPAND);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3, 4}));

        auto reshape_node = a_grad->parents[0];
        ASSERT_NE(reshape_node, nullptr);

        EXPECT_EQ(reshape_node->op, ops::OpEnum::RESHAPE);
        EXPECT_EQ(reshape_node->view.shape, (std::vector<int>{2, 1, 4}));

        const auto& res_attrs = std::get<ReshapeAttrs>(reshape_node->attrs);
        EXPECT_EQ(res_attrs.shape, (std::vector<int>{2, 1, 4}));
    }
}

TEST_F(OpsSumTest, AdjointGlobalReduction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);

        auto z = ops::sum(a, {}, false);

        GradContext ctx;

        auto seed = make_leaf({}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUM);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::EXPAND);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        auto reshape_node = a_grad->parents[0];
        ASSERT_NE(reshape_node, nullptr);

        EXPECT_EQ(reshape_node->op, ops::OpEnum::RESHAPE);
    }
}

TEST_F(OpsSumTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);

        auto z = ops::sum(a, {0}, true);

        GradContext ctx;

        auto seed = make_leaf({1, 3}, false, dev);
        ctx.set(z, seed);

        auto existing_grad = make_leaf({2, 3}, false, dev);
        ctx.set(a, existing_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUM);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsSumTest, AdjointSkipWhenNoGradRequired) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);

        auto z = ops::sum(a, {0}, true);

        GradContext ctx;

        auto seed = make_leaf({1, 3}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUM);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        EXPECT_EQ(ctx.get(a), nullptr);
    }
}