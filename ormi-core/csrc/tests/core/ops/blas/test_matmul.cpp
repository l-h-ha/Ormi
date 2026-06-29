#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "tests/test_utils.h"

#include "core/dtype.h"
#include "core/grad_context.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/node_allocator.h"
#include "core/ops.h"
#include "core/ops/blas/matmul.h"
#include "core/registry.h"

using namespace ormi;

class OpsMatMulTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsMatMulTest, ForwardShapeAndBroadcasting) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({3, 4}, false, dev);
        auto c = ops::matmul(a, b);

        EXPECT_EQ(c->op, ops::OpEnum::MATMUL);
        EXPECT_EQ(c->view.shape, (std::vector<int>{2, 4}));

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
        auto a = make_leaf({2, 3}, false, dev);
        auto b = make_leaf({2, 4}, false, dev);

        EXPECT_THROW(ops::matmul(a, b), std::runtime_error);

        auto c = make_leaf({3}, false, dev);

        EXPECT_THROW(ops::matmul(c, c), std::runtime_error);

        auto batch_a = make_leaf({2, 3, 4}, false, dev);
        auto batch_b = make_leaf({5, 4, 2}, false, dev);

        EXPECT_THROW(ops::matmul(batch_a, batch_b), std::runtime_error);
    }
}

TEST_F(OpsMatMulTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3, 4}, true, dev);
        auto c = ops::matmul(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 4}, false, dev);
        ctx.set(c, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MATMUL);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::MATMUL);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(b_grad->op, ops::OpEnum::MATMUL);
        EXPECT_EQ(b_grad->view.shape, (std::vector<int>{3, 4}));
    }
}

TEST_F(OpsMatMulTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3, 4}, true, dev);
        auto c = ops::matmul(a, b);

        GradContext ctx;

        auto seed = make_leaf({2, 4}, false, dev);
        ctx.set(c, seed);

        auto existing_a_grad = make_leaf({2, 3}, false, dev);
        ctx.set(a, existing_a_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MATMUL);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        EXPECT_EQ(a_grad->parents[0], existing_a_grad);
        EXPECT_EQ(a_grad->parents[1]->op, ops::OpEnum::MATMUL);
    }
}

TEST_F(OpsMatMulTest, AdjointExpandedGradientGenericPath) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto b = make_leaf({3, 4}, true, dev);
        auto c = ops::matmul(a, b);

        auto scalar_grad = make_leaf({1, 1}, false, dev);

        ExpandAttrs exp_attrs{{2, 4}};
        auto dz = std::allocate_shared<Node>(
            get_node_allocator<Node>(),
            ops::OpEnum::EXPAND,
            std::vector<std::shared_ptr<Node>>{scalar_grad},
            exp_attrs
        );

        GradContext ctx;
        ctx.set(c, dz);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::MATMUL);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(c, ctx);

        auto a_grad = ctx.get(a);
        auto b_grad = ctx.get(b);

        ASSERT_NE(a_grad, nullptr);
        EXPECT_EQ(a_grad->op, ops::OpEnum::MATMUL);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3}));

        ASSERT_NE(b_grad, nullptr);
        EXPECT_EQ(b_grad->op, ops::OpEnum::MATMUL);
        EXPECT_EQ(b_grad->view.shape, (std::vector<int>{3, 4}));
    }
}