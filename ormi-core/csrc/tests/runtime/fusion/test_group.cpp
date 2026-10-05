#include <array>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
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

    TEST(FusionCandidateTest, groupsCompatibleSiblingsWithSharedParent) {
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

        const auto candidates = ormi::core::runtime::fusion::find_candidates(roots);

        ASSERT_EQ(candidates.size(), 1u);
        const auto group = candidates.find(negative.get());
        ASSERT_NE(group, candidates.end());
        EXPECT_EQ(
            group->second,
            (std::vector<ormi::core::Tensor*>{
                exponential.get(),
                sine.get()
            })
        );
    }

    TEST(FusionCandidateTest, findsInternalSiblingsFromOneRoot) {
        const auto input = tensor();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        const auto root = ormi::core::ops::add(negative, exponential);
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};

        const auto candidates = ormi::core::runtime::fusion::find_candidates(roots);

        ASSERT_EQ(candidates.size(), 1u);
        const auto group = candidates.find(negative.get());
        ASSERT_NE(group, candidates.end());
        EXPECT_EQ(
            group->second,
            (std::vector<ormi::core::Tensor*>{exponential.get()})
        );
    }

    TEST(FusionCandidateTest, leavesDifferentParentsSeparate) {
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
            ormi::core::runtime::fusion::find_candidates(roots).empty()
        );
    }

    TEST(FusionCandidateTest, leavesDifferentOutputTypesSeparate) {
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
            ormi::core::runtime::fusion::find_candidates(roots).empty()
        );
    }

    TEST(FusionCandidateTest, groupsEachCompatibleOutputType) {
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

        const auto candidates = ormi::core::runtime::fusion::find_candidates(roots);

        ASSERT_EQ(candidates.size(), 2u);
        ASSERT_NE(candidates.find(negative.get()), candidates.end());
        EXPECT_EQ(
            candidates.at(negative.get()),
            (std::vector<ormi::core::Tensor*>{sine.get()})
        );
        ASSERT_NE(candidates.find(first_converted.get()), candidates.end());
        EXPECT_EQ(
            candidates.at(first_converted.get()),
            (std::vector<ormi::core::Tensor*>{second_converted.get()})
        );
    }

    TEST(FusionCandidateTest, discoveryDoesNotDecideEligibility) {
        const auto input = tensor();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto first = ormi::core::ops::reshape(mapped, {4});
        const auto second = ormi::core::ops::reshape(mapped, {4});
        const std::array<ormi::core::Tensor*, 2> roots{
            first.get(),
            second.get()
        };

        const auto candidates = ormi::core::runtime::fusion::find_candidates(roots);

        ASSERT_EQ(candidates.size(), 1u);
        const auto group = candidates.find(first.get());
        ASSERT_NE(group, candidates.end());
        ASSERT_EQ(
            group->second,
            (std::vector<ormi::core::Tensor*>{second.get()})
        );
        EXPECT_FALSE(
            ormi::core::runtime::lower_group(
                *first,
                group->second
            ).has_value()
        );
    }

    TEST(FusionCandidateTest, stopsAtRealizedBoundary) {
        const auto input = tensor();
        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        const auto root = std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<boundary_op>(),
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
        EXPECT_TRUE(ormi::core::runtime::fusion::find_candidates(roots).empty());
    }

}
