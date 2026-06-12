#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <stdexcept>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/ops/blas/matmul.h"

using namespace ormi;

class OpsMatMulTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsMatMulTest, ForwardShapeAndBroadcasting) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({3, 4}, false, dev);
        auto c = ops::matmul(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::MATMUL);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 4}));

        // Batched Broadcasting ( [2, 1, 4, 5] @ [3, 5, 2] -> [2, 3, 4, 2] )
        auto batch_a = make_leaf({2, 1, 4, 5}, false, dev);
        auto batch_b = make_leaf({3, 5, 2}, false, dev);
        auto batch_c = ops::matmul(batch_a, batch_b);
        EXPECT_EQ(batch_c->view.shape, (std::vector<int>{2, 3, 4, 2}));
    }
}

TEST_F(OpsMatMulTest, DeviceMismatchThrows) {
    if (available_devices.size() < 2) {
        GTEST_SKIP() << "Skipping device mismatch test: Only one device available.";
    }

    auto dev1 = available_devices[0];
    auto dev2 = available_devices[1];

    auto a = make_leaf({2, 3}, false, dev1);
    auto b = make_leaf({3, 4}, false, dev2);
    EXPECT_THROW(ops::matmul(a, b), std::runtime_error);
}

TEST_F(OpsMatMulTest, InvalidShapesThrow) {
    for (auto dev : available_devices) {
        // K_a (3) != K_b (2)
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 4}, false, dev); 
        EXPECT_THROW(ops::matmul(a, b), std::runtime_error);

        // vectors not supported in pure matmul
        auto c = make_leaf({3}, false, dev); 
        EXPECT_THROW(ops::matmul(c, c), std::runtime_error);

        // [2] and [5] cannot broadcast
        auto batch_a = make_leaf({2, 3, 4}, false, dev);
        auto batch_b = make_leaf({5, 4, 2}, false, dev);
        EXPECT_THROW(ops::matmul(batch_a, batch_b), std::runtime_error);
    }
}

TEST_F(OpsMatMulTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        // [2, 2, 3] @ [3, 4] -> C: [2, 2, 4]
        auto a = make_leaf({2, 2, 3}, true, dev);
        auto b = make_leaf({3, 4}, true, dev);
        auto c = ops::matmul(a, b);
        c->grad = make_leaf({2, 2, 4}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MATMUL);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(c);

        // dX = dZ @ Y^T
        // dZ: [2, 2, 4] @ Y^T: [4, 3] -> [2, 2, 3]
        // This matches A, so unbroadcast does nothing.
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 2, 3}));
        EXPECT_EQ(a->grad->op, ops::OpEnum::MATMUL); 

        // dY = X^T @ dZ
        // X^T: [2, 3, 2] @ dZ: [2, 2, 4] -> [2, 3, 4]
        // B is [3, 4] -> Unbroadcast MUST insert a SUM node to collapse the batch.
        ASSERT_NE(b->grad, nullptr);
        EXPECT_EQ(b->grad->view.shape, (std::vector<int>{3, 4}));
        EXPECT_EQ(b->grad->op, ops::OpEnum::SUM);
    }
}

TEST_F(OpsMatMulTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3, 4}, true, dev);
        auto c = ops::matmul(a, b);
        
        c->grad = make_leaf({2, 4}, false, dev);

        // pre-seed the gradient on A to simulate a tensor used in multiple ops
        a->grad = make_leaf({2, 3}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MATMUL);
        adjoint_func(c);

        // A->grad should now be an ADD node combining its old gradient with the new one
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}