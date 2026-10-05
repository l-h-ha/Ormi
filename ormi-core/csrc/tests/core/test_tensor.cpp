#include <memory>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
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
