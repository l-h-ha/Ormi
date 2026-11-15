#include <array>
#include <memory>
#include <unordered_set>
#include <utility>
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
    std::shared_ptr<ormi::core::Tensor> tensor(ormi::core::Shape shape = {4}) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
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

    TEST(FusionCandidateTest, recordsSharedScalarsWithoutRealizingThem) {
        const auto input = tensor({});
        const auto shared = ormi::core::ops::mul(input, input);
        const auto left = ormi::core::ops::add(shared, input);
        const auto right = ormi::core::ops::sub(shared, input);
        const auto root = ormi::core::ops::mul(left, right);
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};
        std::unordered_set<const ormi::core::Tensor*> shared_scalars;

        ormi::core::runtime::fusion::find_candidates(roots, &shared_scalars);

        EXPECT_EQ(shared_scalars, (std::unordered_set<const ormi::core::Tensor*>{shared.get()}));
        EXPECT_FALSE(shared->realized());
    }

    TEST(FusionCandidateTest, leavesSharedVectorsOutOfScalarBoundaries) {
        const auto input = tensor();
        const auto shared = ormi::core::ops::mul(input, input);
        const auto root = ormi::core::ops::add(shared, shared);
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};
        std::unordered_set<const ormi::core::Tensor*> shared_scalars;

        ormi::core::runtime::fusion::find_candidates(roots, &shared_scalars);

        EXPECT_TRUE(shared_scalars.empty());
    }

    TEST(FusionCandidateTest, keepsSmallScalarGraphsWithinOneKernel) {
        const auto input = tensor({});
        const auto shared = ormi::core::ops::mul(input, input);
        const auto root = ormi::core::ops::add(shared, shared);
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};
        std::unordered_set<const ormi::core::Tensor*> shared_scalars;

        ormi::core::runtime::fusion::find_candidates(roots, &shared_scalars, 256);

        EXPECT_TRUE(shared_scalars.empty());
    }

    TEST(FusionCandidateTest, keepsWideBatchOfSmallScalarGraphsFused) {
        const auto input = tensor({});
        std::vector<std::shared_ptr<ormi::core::Tensor>> outputs;
        std::vector<ormi::core::Tensor*> roots;

        for (int i = 0; i < 128; ++i) {
            const auto shared = ormi::core::ops::mul(input, input);
            const auto left = ormi::core::ops::add(shared, input);
            const auto right = ormi::core::ops::sub(shared, input);
            outputs.push_back(ormi::core::ops::mul(left, right));
            roots.push_back(outputs.back().get());
        }

        std::unordered_set<const ormi::core::Tensor*> shared_scalars;

        ormi::core::runtime::fusion::find_candidates(roots, &shared_scalars, 256);

        EXPECT_TRUE(shared_scalars.empty());
    }

    TEST(FusionCandidateTest, detectsLargeRootAfterItsDependenciesWereVisited) {
        const auto input = tensor({});
        std::vector<std::shared_ptr<ormi::core::Tensor>> outputs;
        std::vector<ormi::core::Tensor*> roots;

        for (int i = 0; i < 128; ++i) {
            const auto shared = ormi::core::ops::mul(input, input);
            outputs.push_back(ormi::core::ops::add(shared, shared));
            roots.push_back(outputs.back().get());
        }

        auto combined = outputs.front();

        for (std::size_t i = 1; i < outputs.size(); ++i) {
            combined = ormi::core::ops::add(combined, outputs[i]);
        }

        roots.push_back(combined.get());
        std::unordered_set<const ormi::core::Tensor*> shared_scalars;

        ormi::core::runtime::fusion::find_candidates(roots, &shared_scalars, 256);

        EXPECT_FALSE(shared_scalars.empty());
    }

    TEST(FusionCandidateTest, ignoresRealizedSharedScalars) {
        const auto input = tensor({});
        *static_cast<float*>(input->data()) = 0.25f;
        const auto shared = ormi::core::ops::mul(input, input);
        shared->realize();
        const auto root = ormi::core::ops::add(shared, shared);
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};
        std::unordered_set<const ormi::core::Tensor*> shared_scalars;

        ormi::core::runtime::fusion::find_candidates(roots, &shared_scalars);

        EXPECT_TRUE(shared_scalars.empty());
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
        const auto step = ormi::core::runtime::lower(
            *first, candidates, ormi::core::runtime::Preparation::direct
        );

        EXPECT_FALSE(step.allocates());
        EXPECT_EQ(step.output_count(), 1u);
        EXPECT_EQ(&step.input(0), mapped.get());
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
        auto step = ormi::core::runtime::lower(*root, ormi::core::runtime::Preparation::direct);
        step.run();
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};

        EXPECT_TRUE(root->realized());
        EXPECT_FALSE(negative->realized());
        EXPECT_FALSE(exponential->realized());
        EXPECT_TRUE(ormi::core::runtime::fusion::find_candidates(roots).empty());
    }

}
