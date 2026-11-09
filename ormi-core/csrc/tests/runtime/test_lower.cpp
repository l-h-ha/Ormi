#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/group.h"
#include "core/runtime/lower.h"

namespace {
    namespace core = ormi::core;
    using Candidates = core::runtime::fusion::Candidates;
    using Step = core::runtime::Step;
    using Preparation = core::runtime::Preparation;

    const core::Op copy_op{
        .name = "copy",
        .output = core::Output::allocate,
        .parents = 1,
        .compute = [](core::Tensor& output) {
            *static_cast<float*>(output.data()) =
                *static_cast<const float*>(output.parents()[0]->data());
        },
        .adjoint = nullptr,
        .scalar = std::monostate{}
    };

    class LowerTest : public testing::TestWithParam<std::tuple<Preparation, bool>> {
    protected:
        std::shared_ptr<core::Tensor> input = core::ops::full({}, 2.0f, core::Device::cpu());
        std::shared_ptr<core::Tensor> mapped = core::ops::add(input, input);
        std::shared_ptr<core::Tensor> negative = core::ops::neg(mapped);
        std::shared_ptr<core::Tensor> exponential = core::ops::exp(mapped);
        std::shared_ptr<core::Tensor> sine = core::ops::sin(mapped);
        std::array<core::Tensor*, 3> roots{negative.get(), exponential.get(), sine.get()};
        std::unordered_set<const core::Tensor*> boundaries;

        Preparation preparation() const {
            return std::get<0>(GetParam());
        }

        bool bounded() const {
            return std::get<1>(GetParam());
        }

        Step select(core::Tensor& output, const Candidates& candidates) {
            return bounded()
                ? core::runtime::lower(output, candidates, boundaries, preparation())
                : core::runtime::lower(output, candidates, preparation());
        }

        void expect_values() {
            EXPECT_FLOAT_EQ(*static_cast<const float*>(negative->data()), -4.0f);
            EXPECT_NEAR(*static_cast<const float*>(exponential->data()), std::exp(4.0f), 1e-5f);
            EXPECT_NEAR(*static_cast<const float*>(sine->data()), std::sin(4.0f), 1e-6f);
        }
    };

    TEST_P(LowerTest, selectsGroupedFusionBeforeSingleOutput) {
        const auto candidates = core::runtime::fusion::find_candidates(roots);
        auto current = select(*negative, candidates);

        ASSERT_EQ(current.output_count(), 3u);
        EXPECT_EQ(&current.output(0), negative.get());
        EXPECT_EQ(&current.output(1), exponential.get());
        EXPECT_EQ(&current.output(2), sine.get());
        ASSERT_EQ(current.input_count(), 1u);
        EXPECT_EQ(&current.input(0), input.get());
        EXPECT_FALSE(negative->realized());
        EXPECT_FALSE(exponential->realized());
        EXPECT_FALSE(sine->realized());

        current.run();
        expect_values();
        EXPECT_FALSE(mapped->realized());
    }

    TEST_P(LowerTest, lowersSingleOutputWithoutCandidates) {
        auto current = select(*negative, {});

        ASSERT_EQ(current.output_count(), 1u);
        ASSERT_EQ(current.input_count(), 1u);
        EXPECT_EQ(&current.input(0), input.get());
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(negative->data()), -4.0f);
        EXPECT_FALSE(exponential->realized());
    }

    TEST_P(LowerTest, lowersRootWithExplicitPreparation) {
        auto current = core::runtime::lower(*negative, preparation());

        ASSERT_EQ(current.output_count(), 1u);
        ASSERT_EQ(current.input_count(), 1u);
        EXPECT_EQ(&current.input(0), input.get());
        current.prepare();
        current.prepare();
        EXPECT_FALSE(negative->realized());
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(negative->data()), -4.0f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_P(LowerTest, preparesGroupedFusionWithoutRealizingOutputs) {
        const auto candidates = core::runtime::fusion::find_candidates(roots);
        std::vector<Step> current;
        current.push_back(select(*negative, candidates));

        ASSERT_EQ(current.front().output_count(), 3u);
        core::runtime::prepare(current);
        core::runtime::prepare(current);
        EXPECT_FALSE(negative->realized());
        EXPECT_FALSE(exponential->realized());
        EXPECT_FALSE(sine->realized());
        EXPECT_FALSE(mapped->realized());

        current.front().run();
        expect_values();
        *static_cast<float*>(input->data()) = 3.0f;
        current.front().run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(negative->data()), -6.0f);
        EXPECT_NEAR(*static_cast<const float*>(exponential->data()), std::exp(6.0f), 1e-4f);
        EXPECT_NEAR(*static_cast<const float*>(sine->data()), std::sin(6.0f), 1e-6f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_P(LowerTest, ignoresEmptyCandidateGroup) {
        const Candidates candidates{{negative.get(), {}}};
        auto current = select(*negative, candidates);

        EXPECT_EQ(current.output_count(), 1u);
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(negative->data()), -4.0f);
    }

    TEST_P(LowerTest, ignoresCandidatesForAnotherOutput) {
        const Candidates candidates{{exponential.get(), {sine.get()}}};
        auto current = select(*negative, candidates);

        EXPECT_EQ(current.output_count(), 1u);
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(negative->data()), -4.0f);
        EXPECT_FALSE(exponential->realized());
        EXPECT_FALSE(sine->realized());
    }

    TEST_P(LowerTest, fallsBackToSingleFusionForIncompatibleGroup) {
        const auto converted = core::ops::cast(mapped, core::DType::float64);
        const Candidates candidates{{negative.get(), {converted.get()}}};
        auto current = select(*negative, candidates);

        ASSERT_EQ(current.output_count(), 1u);
        ASSERT_EQ(current.input_count(), 1u);
        EXPECT_EQ(&current.input(0), input.get());
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(negative->data()), -4.0f);
        EXPECT_FALSE(converted->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_P(LowerTest, preservesAliasFallbackForUnrepresentableGroup) {
        const auto first = core::ops::reshape(mapped, {1});
        const auto second = core::ops::reshape(mapped, {1});
        const std::array<core::Tensor*, 2> outputs{first.get(), second.get()};
        const auto candidates = core::runtime::fusion::find_candidates(outputs);
        ASSERT_EQ(candidates.size(), 1u);
        auto current = select(*first, candidates);

        EXPECT_FALSE(current.allocates());
        ASSERT_EQ(current.output_count(), 1u);
        ASSERT_EQ(current.input_count(), 1u);
        EXPECT_EQ(&current.input(0), mapped.get());
        mapped->realize();
        current.run();
        EXPECT_EQ(first->buffer(), mapped->buffer());
        EXPECT_FALSE(second->realized());
    }

    TEST_P(LowerTest, preservesComputeFallbackForUnrepresentableGroup) {
        const auto output = std::make_shared<core::Tensor>(
            core::op_ref<copy_op>(),
            std::vector<std::shared_ptr<core::Tensor>>{mapped},
            mapped->device(), mapped->shape(), mapped->dtype()
        );
        const Candidates candidates{{output.get(), {exponential.get()}}};
        auto current = select(*output, candidates);

        EXPECT_TRUE(current.allocates());
        ASSERT_EQ(current.output_count(), 1u);
        ASSERT_EQ(current.input_count(), 1u);
        EXPECT_EQ(&current.input(0), mapped.get());
        mapped->realize();
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 4.0f);
        EXPECT_FALSE(exponential->realized());
    }

    TEST_P(LowerTest, respectsBoundariesIndependentlyOfPreparation) {
        boundaries.insert(mapped.get());
        const auto candidates = core::runtime::fusion::find_candidates(roots);
        auto current = select(*negative, candidates);

        ASSERT_EQ(current.output_count(), 3u);
        ASSERT_EQ(current.input_count(), 1u);
        EXPECT_EQ(&current.input(0), bounded() ? mapped.get() : input.get());
        if (bounded()) {
            mapped->realize();
        }
        current.run();
        expect_values();
        EXPECT_EQ(mapped->realized(), bounded());
    }

    INSTANTIATE_TEST_SUITE_P(
        Execution, LowerTest,
        testing::Combine(
            testing::Values(Preparation::direct, Preparation::deferred),
            testing::Bool()
        ),
        [](const testing::TestParamInfo<std::tuple<Preparation, bool>>& mode) {
            return std::string(std::get<0>(mode.param) == Preparation::direct ? "Direct" : "Deferred") +
                (std::get<1>(mode.param) ? "Bounded" : "Unbounded");
        }
    );
}
