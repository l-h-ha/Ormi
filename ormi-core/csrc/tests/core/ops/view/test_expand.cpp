#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/attributes.h"
#include "core/ops/view/expand.h"

using namespace ormi;

class OpsExpandTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsExpandTest, ZeroCopyStrideInference) {
    for (auto dev : available_devices) {
        // Prepending new dimensions
        // Base shape: [3, 4], Contiguous strides: [4, 1]
        auto a = make_leaf({3, 4}, false, dev);
        auto b = ops::expand(a, {2, 3, 4});

        EXPECT_EQ(b->op, ops::OpEnum::EXPAND);
        EXPECT_EQ(b->view.shape, (std::vector<int>{2, 3, 4}));
        
        // prepended dimension MUST have a stride of 0
        EXPECT_EQ(b->view.strides, (std::vector<int>{0, 4, 1}));

        // Stretching inner dimensions
        // Base shape: [2, 1, 4], Contiguous strides: [4, 4, 1]
        auto c = make_leaf({2, 1, 4}, false, dev);
        auto d = ops::expand(c, {2, 5, 4});

        EXPECT_EQ(d->view.shape, (std::vector<int>{2, 5, 4}));
        
        // inner dimension (size 1 -> 5) MUST have its stride nullified to 0
        EXPECT_EQ(d->view.strides, (std::vector<int>{4, 0, 1}));
    }
}

TEST_F(OpsExpandTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        // Base shape [1, 4] expanded to [3, 4]
        auto a = make_leaf({1, 4}, true, dev);
        auto z = ops::expand(a, {3, 4});

        // Seed gradient
        z->grad = make_leaf({3, 4}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::EXPAND);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(z);

        // backpropagation through an EXPAND requires collapsing the gradient
        // using unbroadcast(). This creates a SUM node across the expanded axes.
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::SUM);
        
        // The collapsed gradient must match the original pre-expanded shape
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{1, 4}));
        
        // Verify the reduce axes logic inside the SUM node
        const auto& attrs = std::get<ReduceAttrs>(a->grad->attrs);
        EXPECT_EQ(attrs.axes, (std::vector<int>{0})); // Axis 0 was stretched from 1 to 3
    }
}

TEST_F(OpsExpandTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({1, 4}, true, dev);
        auto z = ops::expand(a, {3, 4});
        
        z->grad = make_leaf({3, 4}, false, dev);
        a->grad = make_leaf({1, 4}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::EXPAND);
        adjoint_func(z);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{1, 4}));
    }
}