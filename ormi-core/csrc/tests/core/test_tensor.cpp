#include <limits>
#include <memory>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    TEST(TensorTest, leafStartsRealized) {
        const ormi::core::Tensor tensor(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3},
            ormi::core::DType::float32
        );

        EXPECT_TRUE(tensor.realized());
        EXPECT_EQ(tensor.op(), nullptr);
        EXPECT_TRUE(tensor.parents().empty());
        EXPECT_EQ(tensor.graph_depth(), 0);
    }

    TEST(TensorTest, tracksLongestParentChainWithoutRealizing) {
        const auto leaf = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{}, ormi::core::DType::float64
        );
        const auto first = ormi::core::ops::neg(leaf);
        const auto second = ormi::core::ops::mul(first, leaf);
        const auto third = ormi::core::ops::add(second, first);

        EXPECT_EQ(leaf->graph_depth(), 0);
        EXPECT_EQ(first->graph_depth(), 1);
        EXPECT_EQ(second->graph_depth(), 2);
        EXPECT_EQ(third->graph_depth(), 3);
        EXPECT_FALSE(third->realized());
    }

    TEST(TensorTest, rejectsOverflowingStridesForEmptyTensor) {
        const auto limit = std::numeric_limits<std::int64_t>::max();

        EXPECT_THROW(
            (void)ormi::core::Tensor(ormi::core::Device::cpu(), {0, limit, 2}, ormi::core::DType::float32),
            ormi::ShapeError
        );
    }

    TEST(TensorTest, constructsEmptyTensorAfterLargeDimensions) {
        const auto limit = std::numeric_limits<std::int64_t>::max();
        const ormi::core::Tensor tensor(
            ormi::core::Device::cpu(), {limit, limit, 0}, ormi::core::DType::float32
        );

        EXPECT_EQ(tensor.numel(), 0u);
        EXPECT_EQ(tensor.bytes(), 0u);
        EXPECT_TRUE(tensor.is_contiguous());
    }

    TEST(TensorTest, aliasOperationReusesParentBuffer) {
        static const ormi::core::Op alias_op{
            .name = "test_alias",
            .output = ormi::core::Output::alias,
            .parents = 1,
            .compute = nullptr,
            .adjoint = nullptr,
            .scalar = std::monostate{}
        };
        const auto parent = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        ormi::core::Tensor alias(
            ormi::core::op_ref<alias_op>(),
            std::monostate{},
            parent,
            parent->shape(),
            std::vector<std::int64_t>{1},
            0
        );

        EXPECT_FALSE(alias.realized());
        EXPECT_EQ(alias.op(), &alias_op);
        EXPECT_EQ(alias.graph_depth(), 1);
        alias.realize();

        EXPECT_TRUE(alias.realized());
        EXPECT_EQ(alias.buffer(), parent->buffer());
    }

    TEST(TensorTest, rejectsMalformedOperationDefinition) {
        static const ormi::core::Op malformed{
            .name = "malformed",
            .output = ormi::core::Output::allocate,
            .parents = 1,
            .compute = nullptr,
            .adjoint = nullptr,
            .scalar = std::monostate{}
        };
        const auto parent = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float32
        );

        EXPECT_THROW(
            (void)ormi::core::Tensor(
                ormi::core::op_ref<malformed>(),
                std::vector<std::shared_ptr<ormi::core::Tensor>>{parent},
                parent->device(),
                parent->shape(),
                parent->dtype()
            ),
            ormi::OperationError
        );
    }
}
