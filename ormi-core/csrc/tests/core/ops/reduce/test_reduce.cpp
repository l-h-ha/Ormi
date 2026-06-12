#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <algorithm>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/attributes.h"
#include "core/ops/reduce/sum.h"

using namespace ormi;

class OpsSumTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsSumTest, GlobalReductionShape) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);

        // Global reduction (empty axes), keepdims = false -> Scalar
        auto clean_sum = ops::sum(a, {}, false);
        EXPECT_EQ(clean_sum->op, ops::OpEnum::SUM);
        EXPECT_TRUE(clean_sum->view.shape.empty());

        // Global reduction, keepdims = true -> Retains dimensions as 1s ({1, 1, 1})
        auto keep_sum = ops::sum(a, {}, true);
        EXPECT_EQ(keep_sum->view.shape, (std::vector<int>{1, 1, 1}));
    }
}

TEST_F(OpsSumTest, AxisSpecificReductionShape) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);

        // Reduce single axis (axis 1), keepdims = false -> {2, 4}
        auto sum_ax1 = ops::sum(a, {1}, false);
        EXPECT_EQ(sum_ax1->view.shape, (std::vector<int>{2, 4}));

        // Reduce single axis (axis 1), keepdims = true -> {2, 1, 4}
        auto sum_ax1_keep = ops::sum(a, {1}, true);
        EXPECT_EQ(sum_ax1_keep->view.shape, (std::vector<int>{2, 1, 4}));

        // Reduce multiple axes (axes 0 and 2), keepdims = false -> {3}
        auto sum_ax02 = ops::sum(a, {0, 2}, false);
        EXPECT_EQ(sum_ax02->view.shape, (std::vector<int>{3}));

        // Reduce multiple axes (axes 0 and 2), keepdims = true -> {1, 3, 1}
        auto sum_ax02_keep = ops::sum(a, {0, 2}, true);
        EXPECT_EQ(sum_ax02_keep->view.shape, (std::vector<int>{1, 3, 1}));
    }
}

TEST_F(OpsSumTest, AdjointWithKeepDims) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);
        
        // Sum along axis 1, keeping dimensions -> Output shape is {2, 1, 4}
        auto z = ops::sum(a, {1}, true);
        
        // Seed the gradient
        z->grad = make_leaf({2, 1, 4}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUM);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(z);

        // Since keepdims = true, the gradient graph should jump straight to an EXPAND node
        // to broadcast the 1 back out to the original leaf shape {2, 3, 4}.
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::EXPAND);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3, 4}));

        // Verify that the target shape parameter inside ExpandAttrs matches the leaf
        const auto& exp_attrs = std::get<ExpandAttrs>(a->grad->attrs);
        EXPECT_EQ(exp_attrs.shape, (std::vector<int>{2, 3, 4}));
    }
}

TEST_F(OpsSumTest, AdjointWithoutKeepDims) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);
        
        // Sum along axis 1, dropping dimensions -> Output shape is {2, 4}
        auto z = ops::sum(a, {1}, false);
        
        // Seed the gradient with a match for the collapsed output shape
        z->grad = make_leaf({2, 4}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUM);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(z);

        // When keepdims = false, a RESHAPE node must be created first to un-squeeze 
        // the dimension (expanding {2, 4} to {2, 1, 4}) before feeding into the EXPAND wrapper.
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::EXPAND);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3, 4}));

        // Inspect the parent of the EXPAND node
        auto reshape_node = a->grad->parents[0];
        ASSERT_NE(reshape_node, nullptr);
        EXPECT_EQ(reshape_node->op, ops::OpEnum::RESHAPE);
        EXPECT_EQ(reshape_node->view.shape, (std::vector<int>{2, 1, 4}));

        const auto& res_attrs = std::get<ReshapeAttrs>(reshape_node->attrs);
        EXPECT_EQ(res_attrs.shape, (std::vector<int>{2, 1, 4}));
    }
}

TEST_F(OpsSumTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto z = ops::sum(a, {0}, true); // shape {1, 3}
        
        z->grad = make_leaf({1, 3}, false, dev);

        // Pre-populate a's gradient to ensure clean graph accumulation
        a->grad = make_leaf({2, 3}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SUM);
        adjoint_func(z);

        // The top node under a->grad should now be an ADD operation
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}