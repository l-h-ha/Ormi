#include <array>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/group.h"
#include "core/runtime/lower.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> tensor() {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float32
        );
    }

    void boundary(ormi::core::Tensor&) {}

    const ormi::core::Op boundary_op{
        .name = "boundary",
        .output = ormi::core::Output::allocate,
        .parents = 2,
        .compute = boundary,
        .adjoint = nullptr,
        .scalar = std::monostate{}
    };

    TEST(FusionGroupTest, groupsCompatibleSiblingsWithSharedParent) {
        const auto input = tensor();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        const auto sine = ormi::core::ops::sin(mapped);
        const std::array<ormi::core::Tensor*, 3> roots{
            negative.get(),
            exponential.get(),
            sine.get()
        };

        const auto grouped = ormi::core::runtime::fusion::groups(roots);

        ASSERT_EQ(grouped.size(), 1u);
        const auto group = grouped.find(negative.get());
        ASSERT_NE(group, grouped.end());
        EXPECT_EQ(
            group->second,
            (std::vector<ormi::core::Tensor*>{
                exponential.get(),
                sine.get()
            })
        );
    }

    TEST(FusionGroupTest, findsInternalSiblingsFromOneRoot) {
        const auto input = tensor();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        const auto root = ormi::core::ops::add(negative, exponential);
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};

        const auto grouped = ormi::core::runtime::fusion::groups(roots);

        ASSERT_EQ(grouped.size(), 1u);
        const auto group = grouped.find(negative.get());
        ASSERT_NE(group, grouped.end());
        EXPECT_EQ(
            group->second,
            (std::vector<ormi::core::Tensor*>{exponential.get()})
        );
    }

    TEST(FusionGroupTest, leavesDifferentParentsSeparate) {
        const auto first = tensor();
        const auto second = tensor();
        const auto negative = ormi::core::ops::neg(
            ormi::core::ops::add(first, first)
        );
        const auto exponential = ormi::core::ops::exp(
            ormi::core::ops::add(second, second)
        );
        const std::array<ormi::core::Tensor*, 2> roots{
            negative.get(),
            exponential.get()
        };

        EXPECT_TRUE(
            ormi::core::runtime::fusion::groups(roots).empty()
        );
    }

    TEST(FusionGroupTest, leavesDifferentOutputTypesSeparate) {
        const auto input = tensor();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto converted = ormi::core::ops::cast(
            mapped,
            ormi::core::DType::float64
        );
        const std::array<ormi::core::Tensor*, 2> roots{
            negative.get(),
            converted.get()
        };

        EXPECT_TRUE(
            ormi::core::runtime::fusion::groups(roots).empty()
        );
    }

    TEST(FusionGroupTest, groupsEachCompatibleOutputType) {
        const auto input = tensor();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto first_converted = ormi::core::ops::cast(
            mapped,
            ormi::core::DType::float64
        );
        const auto sine = ormi::core::ops::sin(mapped);
        const auto second_converted = ormi::core::ops::cast(
            mapped,
            ormi::core::DType::float64
        );
        const std::array<ormi::core::Tensor*, 4> roots{
            negative.get(),
            first_converted.get(),
            sine.get(),
            second_converted.get()
        };

        const auto grouped = ormi::core::runtime::fusion::groups(roots);

        ASSERT_EQ(grouped.size(), 2u);
        ASSERT_NE(grouped.find(negative.get()), grouped.end());
        EXPECT_EQ(
            grouped.at(negative.get()),
            (std::vector<ormi::core::Tensor*>{sine.get()})
        );
        ASSERT_NE(grouped.find(first_converted.get()), grouped.end());
        EXPECT_EQ(
            grouped.at(first_converted.get()),
            (std::vector<ormi::core::Tensor*>{second_converted.get()})
        );
    }

    TEST(FusionGroupTest, stopsAtRealizedBoundary) {
        const auto input = tensor();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        const auto root = std::make_shared<ormi::core::Tensor>(
            boundary_op,
            std::vector<std::shared_ptr<ormi::core::Tensor>>{
                negative,
                exponential
            },
            input->device(),
            input->shape(),
            input->dtype()
        );
        auto step = ormi::core::runtime::lower(*root);
        step.run();
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};

        EXPECT_TRUE(root->realized());
        EXPECT_FALSE(negative->realized());
        EXPECT_FALSE(exponential->realized());
        EXPECT_TRUE(ormi::core::runtime::fusion::groups(roots).empty());
    }

}
