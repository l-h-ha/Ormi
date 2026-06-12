#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/log.h"

using namespace ormi;

class OpsLogTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
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

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::LOG);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        
        // Since shapes match, unbroadcast returns the DIV node
        EXPECT_EQ(a->grad->op, ops::OpEnum::DIV);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        // Verify the chain: DIV(dz, x)
        auto dz = a->grad->parents[0];
        auto x = a->grad->parents[1];
        
        EXPECT_EQ(dz, c->grad);
        EXPECT_EQ(x, a);
    }
}

TEST_F(OpsLogTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::log(a);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Pre-populate input grad to trigger accumulation
        a->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::LOG);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}