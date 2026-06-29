#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/test_utils.h"

#include "core/attributes.h"
#include "core/dtype.h"
#include "core/grad_context.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/ops/view/reshape.h"
#include "core/ops/view/transpose.h"
#include "core/registry.h"

using namespace ormi;

class OpsReshapeTest : public ormi::testing::HardwareTestSuite {
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

TEST_F(OpsReshapeTest, ContiguousReshapeView) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);

        auto b = ops::reshape(a, {2, 3, 2, 2});
        EXPECT_EQ(b->view.shape, (std::vector<int>{2, 3, 2, 2}));
        EXPECT_EQ(b->view.strides, (std::vector<int>{12, 4, 2, 1}));

        auto c = ops::reshape(a, {6, 4});
        EXPECT_EQ(c->view.shape, (std::vector<int>{6, 4}));
        EXPECT_EQ(c->view.strides, (std::vector<int>{4, 1}));
    }
}

TEST_F(OpsReshapeTest, NonContiguousReshapeThrows) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);

        std::vector<int> axes = {0, 2, 1};
        auto transposed = ops::transpose(a, axes);

        EXPECT_THROW(ops::reshape(transposed, {2, 12}), std::runtime_error);
    }
}

TEST_F(OpsReshapeTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);
        auto z = ops::reshape(a, {6, 4});

        GradContext ctx;

        auto seed = make_leaf({6, 4}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::RESHAPE);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::RESHAPE);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3, 4}));

        const auto& res_attrs = std::get<ReshapeAttrs>(a_grad->attrs);
        EXPECT_EQ(res_attrs.shape, (std::vector<int>{2, 3, 4}));
    }
}

TEST_F(OpsReshapeTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);
        auto z = ops::reshape(a, {6, 4});

        GradContext ctx;

        auto seed = make_leaf({6, 4}, false, dev);
        ctx.set(z, seed);

        auto existing_grad = make_leaf({2, 3, 4}, false, dev);
        ctx.set(a, existing_grad);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::RESHAPE);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        auto a_grad = ctx.get(a);
        ASSERT_NE(a_grad, nullptr);

        EXPECT_EQ(a_grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a_grad->view.shape, (std::vector<int>{2, 3, 4}));
    }
}

TEST_F(OpsReshapeTest, AdjointSkipWhenNoGradRequired) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);
        auto z = ops::reshape(a, {6, 4});

        GradContext ctx;

        auto seed = make_leaf({6, 4}, false, dev);
        ctx.set(z, seed);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::RESHAPE);
        ASSERT_NE(adjoint_func, nullptr);

        adjoint_func(z, ctx);

        EXPECT_EQ(ctx.get(a), nullptr);
    }
}