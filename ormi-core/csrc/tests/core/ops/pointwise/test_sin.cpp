#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/sin.h"

using namespace ormi;

class OpsSinTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsSinTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto c = ops::sin(a);

        EXPECT_EQ(c->op, ops::OpEnum::SIN);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsSinTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::sin(a);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SIN);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));

        // Verify the chain: MUL(COS(x), dz)
        auto cos_x = a->grad->parents[0];
        auto dz = a->grad->parents[1];
        
        EXPECT_EQ(dz, c->grad);
        EXPECT_EQ(cos_x->op, ops::OpEnum::COS);
        EXPECT_EQ(cos_x->parents[0], a);
    }
}

TEST_F(OpsSinTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto c = ops::sin(a);

        // Seed gradient
        c->grad = make_leaf({2, 3}, false, dev);

        // Pre-populate input grad
        a->grad = make_leaf({2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::SIN);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}