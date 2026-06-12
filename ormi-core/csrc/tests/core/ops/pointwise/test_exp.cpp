#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/exp.h"

using namespace ormi;

class OpsExpTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsExpTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto c = ops::exp(a);

        EXPECT_EQ(c->op, ops::OpEnum::EXP);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsExpTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::exp(a);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::EXP);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        // Verify the chain: MUL(node, dz)
        auto exp_node = a->grad->parents[0];
        auto dz = a->grad->parents[1];
        
        EXPECT_EQ(exp_node, c); // Reuses the forward output node
        EXPECT_EQ(dz, c->grad);
    }
}

TEST_F(OpsExpTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::exp(a);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Pre-populate input grad
        a->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::EXP);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}