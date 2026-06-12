#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/pointwise/pow.h" 

using namespace ormi;

class OpsPowTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsPowTest, ForwardShapeInference) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::pow(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::POW);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsPowTest, AdjointGraphConstruction_BaseGrad) {
    for (auto dev : available_devices) {
        // Base requires grad, exponent does not
        auto a = make_leaf({2, 3}, true, dev); 
        auto b = make_leaf({2, 3}, false, dev);
        auto c = ops::pow(a, b);

        c->grad = make_leaf({2, 3}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::POW);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(b->grad, nullptr); // Exponent shouldn't receive gradients
        
        // Outermost node of local_dx is MUL(dz, y * x^(y-1))
        EXPECT_EQ(a->grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsPowTest, AdjointGraphConstruction_ExpGrad) {
    for (auto dev : available_devices) {
        // Exponent requires grad, base does not
        auto a = make_leaf({2, 3}, false, dev); 
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::pow(a, b);

        c->grad = make_leaf({2, 3}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::POW);
        adjoint_func(c);

        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(a->grad, nullptr); // Base shouldn't receive gradients
        
        // Outermost node of local_dy is MUL(dz, node * ln(x))
        EXPECT_EQ(b->grad->op, ops::OpEnum::MUL);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{2, 3}));
    }
}

TEST_F(OpsPowTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        // Both require grad
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({2, 3}, true, dev);
        auto c = ops::pow(a, b);

        c->grad = make_leaf({2, 3}, false, dev);

        // Pre-populate input grads to trigger accumulation branches
        a->grad = make_leaf({2, 3}, false, dev);
        b->grad = make_leaf({2, 3}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::POW);
        adjoint_func(c);

        // Verify accumulation via ADD node
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->op, ops::OpEnum::ADD);
    }
}