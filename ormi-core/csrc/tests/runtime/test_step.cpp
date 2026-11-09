#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/step.h"

namespace {
    namespace core = ormi::core;
    using Step = core::runtime::Step;
    using Expr = core::runtime::fusion::Expr;

    void write_group(core::Tensor& output, std::span<core::Tensor* const> auxiliary, float value) {
        if (value < 0.0f) {
            ORMI_THROW(ormi::OperationError, "Grouped execution failed.");
        }
        *static_cast<float*>(output.data()) = value;
        for (std::size_t index = 0; index < auxiliary.size(); ++index) {
            *static_cast<float*>(auxiliary[index]->data()) =
                value + static_cast<float>(index + 1);
        }
    }

    void run_group(core::Tensor& output, std::span<core::Tensor* const> auxiliary, const Expr& expression) {
        write_group(
            output,
            auxiliary,
            *static_cast<const float*>(expression.inputs.front()->data())
        );
    }

    void unexpected_fallback(core::Tensor&, std::span<core::Tensor* const>, const Expr&) {
        ORMI_THROW(ormi::OperationError, "Prepared execution used the fallback.");
    }

    struct PreparedGroup {
        core::Tensor* input;
        std::vector<core::Tensor*> auxiliary;
        std::size_t* drops;

        ~PreparedGroup() {
            if (drops) {
                ++*drops;
            }
        }
    };

    core::Setup prepare_group(
        core::Tensor&,
        std::span<core::Tensor* const> auxiliary,
        const Expr& expression,
        std::size_t* drops
    ) {
        return core::Setup(
            new PreparedGroup{
                expression.inputs.front(),
                {auxiliary.begin(), auxiliary.end()},
                drops
            },
            [](void* state, core::Tensor& output) {
                const auto& group = *static_cast<PreparedGroup*>(state);
                write_group(
                    output,
                    group.auxiliary,
                    *static_cast<const float*>(group.input->data())
                );
            },
            [](void* state) noexcept {
                delete static_cast<PreparedGroup*>(state);
            }
        );
    }

    core::Setup prepare_group(core::Tensor& output, std::span<core::Tensor* const> auxiliary, const Expr& expression) {
        return prepare_group(output, auxiliary, expression, nullptr);
    }

    class StepTest : public testing::TestWithParam<bool> {
    protected:
        std::shared_ptr<core::Tensor> input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{}, core::DType::float32
        );
        std::shared_ptr<core::Tensor> first = core::ops::neg(input);
        std::shared_ptr<core::Tensor> second = core::ops::neg(input);
        std::shared_ptr<core::Tensor> third = core::ops::neg(input);

        void value(float next) {
            *static_cast<float*>(input->data()) = next;
        }

        Step step() {
            Expr expression{};
            expression.dtype = core::DType::float32;
            expression.elements = 1;
            expression.inputs.push_back(input.get());
            if (GetParam()) {
                return Step::deferred_fusion(
                    *first, prepare_group, nullptr, unexpected_fallback,
                    std::move(expression), {second.get(), third.get()}
                );
            }
            return Step::fusion(
                *first, run_group, std::move(expression),
                {second.get(), third.get()}
            );
        }

        void expect_values(float expected) {
            EXPECT_FLOAT_EQ(*static_cast<const float*>(first->data()), expected);
            EXPECT_FLOAT_EQ(*static_cast<const float*>(second->data()), expected + 1.0f);
            EXPECT_FLOAT_EQ(*static_cast<const float*>(third->data()), expected + 2.0f);
        }
    };

    TEST_P(StepTest, reusesGroupedOutputBuffers) {
        value(2.0f);
        auto current = step();
        current.run();
        expect_values(2.0f);
        const auto first_buffer = first->buffer();
        const auto second_buffer = second->buffer();
        const auto third_buffer = third->buffer();

        value(5.0f);
        current.run();
        expect_values(5.0f);
        EXPECT_EQ(first->buffer(), first_buffer);
        EXPECT_EQ(second->buffer(), second_buffer);
        EXPECT_EQ(third->buffer(), third_buffer);
    }

    TEST_P(StepTest, releasesNewGroupedBuffersAfterFailureAndRetries) {
        value(-1.0f);
        auto current = step();
        EXPECT_THROW(current.run(), ormi::OperationError);
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(second->realized());
        EXPECT_FALSE(third->realized());

        value(3.0f);
        current.run();
        expect_values(3.0f);
    }

    TEST_P(StepTest, preservesExistingGroupedBuffersAfterFailure) {
        const auto initialize = [](core::Tensor& output) {
            *static_cast<float*>(output.data()) = 11.0f;
        };
        auto initialize_first = Step::compute(*first, initialize);
        auto initialize_third = Step::compute(*third, initialize);
        initialize_first.run();
        initialize_third.run();
        const auto first_buffer = first->buffer();
        const auto third_buffer = third->buffer();

        value(-1.0f);
        auto current = step();
        EXPECT_THROW(current.run(), ormi::OperationError);
        EXPECT_FALSE(second->realized());
        EXPECT_EQ(first->buffer(), first_buffer);
        EXPECT_EQ(third->buffer(), third_buffer);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(first->data()), 11.0f);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(third->data()), 11.0f);

        value(4.0f);
        current.run();
        expect_values(4.0f);
        EXPECT_EQ(first->buffer(), first_buffer);
        EXPECT_EQ(third->buffer(), third_buffer);
    }

    TEST_P(StepTest, preservesReadyGroupedBuffersAfterFailure) {
        value(2.0f);
        auto current = step();
        current.run();
        const auto first_buffer = first->buffer();
        const auto second_buffer = second->buffer();
        const auto third_buffer = third->buffer();

        value(-1.0f);
        EXPECT_THROW(current.run(), ormi::OperationError);
        EXPECT_EQ(first->buffer(), first_buffer);
        EXPECT_EQ(second->buffer(), second_buffer);
        EXPECT_EQ(third->buffer(), third_buffer);
        expect_values(2.0f);

        value(6.0f);
        current.run();
        expect_values(6.0f);
    }

    TEST_P(StepTest, releasesEarlierAllocationsWhenALaterOutputSizeOverflows) {
        third = core::ops::neg(core::ops::expand(
            input, {std::numeric_limits<std::int64_t>::max()}
        ));
        auto current = step();

        EXPECT_THROW(current.run(), ormi::ShapeError);
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(second->realized());
        EXPECT_FALSE(third->realized());
        EXPECT_TRUE(input->realized());
    }

    TEST_P(StepTest, preservesExistingBuffersWhenALaterOutputSizeOverflows) {
        auto initialize = Step::compute(*third, [](core::Tensor& output) {
            *static_cast<float*>(output.data()) = 11.0f;
        });
        initialize.run();
        const auto existing = third->buffer();
        second = core::ops::neg(core::ops::expand(
            input, {std::numeric_limits<std::int64_t>::max()}
        ));
        auto current = step();

        EXPECT_THROW(current.run(), ormi::ShapeError);
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(second->realized());
        EXPECT_EQ(third->buffer(), existing);
        EXPECT_FLOAT_EQ(*static_cast<const float*>(third->data()), 11.0f);
    }

    INSTANTIATE_TEST_SUITE_P(
        Fusion, StepTest, testing::Bool(),
        [](const testing::TestParamInfo<bool>& mode) {
            return mode.param ? "Prepared" : "Direct";
        }
    );

    class StepPreparationTest : public testing::TestWithParam<bool> {
    protected:
        static inline std::size_t calls = 0;
        static inline std::size_t batches = 0;
        static inline std::size_t fail_on_call = 0;
        static inline std::size_t drops = 0;
        static inline bool fallback = false;

        std::shared_ptr<core::Tensor> input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{}, core::DType::float32
        );
        std::shared_ptr<core::Tensor> first = core::ops::neg(input);
        std::shared_ptr<core::Tensor> second = core::ops::neg(input);

        void SetUp() override {
            calls = 0;
            batches = 0;
            fail_on_call = 0;
            drops = 0;
            fallback = false;
            value(2.0f);
        }

        void value(float next) {
            *static_cast<float*>(input->data()) = next;
        }

        static core::Setup prepare_one(core::Tensor& output, std::span<core::Tensor* const> auxiliary, const Expr& expression) {
            if (++calls == fail_on_call) {
                ORMI_THROW(ormi::OperationError, "Step preparation failed.");
            }
            return fallback ? core::Setup{} : prepare_group(output, auxiliary, expression, &drops);
        }

        static void prepare_batch(std::span<core::runtime::FusionJob> jobs) {
            ++batches;
            for (const auto& job : jobs) {
                *job.setup = prepare_one(
                    *job.output, {job.auxiliary_outputs, job.auxiliary_count}, *job.expression
                );
            }
        }

        Step step(core::Tensor& output) {
            Expr expression{};
            expression.dtype = core::DType::float32;
            expression.elements = 1;
            expression.inputs.push_back(input.get());
            return Step::deferred_fusion(
                output, prepare_one, GetParam() ? prepare_batch : nullptr,
                run_group, std::move(expression)
            );
        }

        std::vector<Step> steps() {
            std::vector<Step> result;
            result.reserve(2);
            result.push_back(step(*first));
            result.push_back(step(*second));
            return result;
        }

        void expect_values(float expected) {
            EXPECT_FLOAT_EQ(*static_cast<const float*>(first->data()), expected);
            EXPECT_FLOAT_EQ(*static_cast<const float*>(second->data()), expected);
        }
    };

    TEST_P(StepPreparationTest, preparesOnceWithoutAllocatingOutputs) {
        auto current = steps();
        core::runtime::prepare(current);
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(second->realized());
        EXPECT_EQ(calls, 2);

        core::runtime::prepare(current);
        for (auto& item : current) {
            item.run();
        }
        expect_values(2.0f);

        value(5.0f);
        for (auto& item : current) {
            item.run();
        }
        expect_values(5.0f);
        EXPECT_EQ(calls, 2);
        EXPECT_EQ(batches, GetParam() ? 1 : 0);
    }

    TEST_P(StepPreparationTest, completesPreparationWhenSetupIsEmpty) {
        fallback = true;
        auto current = steps();
        core::runtime::prepare(current);
        core::runtime::prepare(current);
        for (auto& item : current) {
            item.run();
        }
        expect_values(2.0f);

        value(4.0f);
        for (auto& item : current) {
            item.run();
        }
        expect_values(4.0f);
        EXPECT_EQ(calls, 2);
        EXPECT_EQ(batches, GetParam() ? 1 : 0);
    }

    TEST_P(StepPreparationTest, retriesPartiallyPreparedSteps) {
        fail_on_call = 2;
        auto current = steps();
        EXPECT_THROW(core::runtime::prepare(current), ormi::OperationError);
        EXPECT_EQ(calls, 2);
        EXPECT_FALSE(first->realized());
        EXPECT_FALSE(second->realized());

        core::runtime::prepare(current);
        for (auto& item : current) {
            item.run();
        }
        expect_values(2.0f);
        EXPECT_EQ(calls, GetParam() ? 4 : 3);
        EXPECT_EQ(batches, GetParam() ? 2 : 0);
    }

    TEST_P(StepPreparationTest, retriesPreparationFailureDuringRun) {
        fail_on_call = 1;
        auto current = step(*first);
        EXPECT_THROW(current.run(), ormi::OperationError);
        EXPECT_FALSE(first->realized());

        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(first->data()), 2.0f);
        EXPECT_EQ(calls, 2);
        EXPECT_EQ(batches, 0);
    }

    TEST_P(StepPreparationTest, releasesPartiallyPreparedStateWhenAbandoned) {
        fail_on_call = 2;
        auto current = steps();
        EXPECT_THROW(core::runtime::prepare(current), ormi::OperationError);
        EXPECT_EQ(drops, 0);

        current.clear();
        EXPECT_EQ(drops, 1);
    }

    TEST_P(StepPreparationTest, releasesReplacedStateWhenFailedPreparationIsRetried) {
        fail_on_call = 2;
        auto current = steps();
        EXPECT_THROW(core::runtime::prepare(current), ormi::OperationError);
        EXPECT_EQ(drops, 0);

        core::runtime::prepare(current);
        EXPECT_EQ(drops, GetParam() ? 1 : 0);
        current.clear();
        EXPECT_EQ(drops, GetParam() ? 3 : 2);
    }

    TEST_P(StepPreparationTest, retriesFailedPreparationWithDirectFallback) {
        fail_on_call = 2;
        auto current = steps();
        EXPECT_THROW(core::runtime::prepare(current), ormi::OperationError);
        fallback = true;

        core::runtime::prepare(current);
        EXPECT_EQ(drops, GetParam() ? 1 : 0);
        for (auto& item : current) {
            item.run();
        }
        expect_values(2.0f);
        current.clear();
        EXPECT_EQ(drops, 1);
    }

    TEST_P(StepPreparationTest, preservesPreparationAfterExecutionFailure) {
        auto current = steps();
        core::runtime::prepare(current);
        value(-1.0f);
        EXPECT_THROW(current[0].run(), ormi::OperationError);
        EXPECT_FALSE(first->realized());

        value(3.0f);
        for (auto& item : current) {
            item.run();
        }
        expect_values(3.0f);
        EXPECT_EQ(calls, 2);
        EXPECT_EQ(batches, GetParam() ? 1 : 0);
    }

    INSTANTIATE_TEST_SUITE_P(
        Fusion, StepPreparationTest, testing::Bool(),
        [](const testing::TestParamInfo<bool>& mode) {
            return mode.param ? "Batch" : "Individual";
        }
    );

    TEST(StepExecutionTest, usesPreparedComputeInsteadOfDirectCompute) {
        const auto input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{}, core::DType::float32
        );
        const auto output = core::ops::neg(input);
        Expr expression{};
        expression.inputs.push_back(input.get());
        auto current = Step::compute(
            *output,
            [](core::Tensor&) {
                ORMI_THROW(ormi::OperationError, "Prepared compute used the fallback.");
            },
            prepare_group(*output, {}, expression)
        );

        *static_cast<float*>(input->data()) = 7.0f;
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 7.0f);
        *static_cast<float*>(input->data()) = 9.0f;
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 9.0f);
    }

    TEST(StepExecutionTest, usesDirectFusionWhenPrepareHookIsAbsent) {
        const auto input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{}, core::DType::float32
        );
        const auto output = core::ops::neg(input);
        Expr expression{};
        expression.inputs.push_back(input.get());
        auto current = Step::deferred_fusion(
            *output, nullptr, nullptr, run_group, std::move(expression)
        );

        current.prepare();
        *static_cast<float*>(input->data()) = 6.0f;
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 6.0f);
        *static_cast<float*>(input->data()) = 8.0f;
        current.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 8.0f);
    }

    TEST(StepExecutionTest, usesDirectFallbackAfterBatchFailureWithoutIndividualPrepare) {
        static std::size_t drops = 0;
        drops = 0;
        const auto input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{}, core::DType::float32
        );
        const auto output = core::ops::neg(input);
        *static_cast<float*>(input->data()) = 2.0f;
        Expr expression{};
        expression.inputs.push_back(input.get());
        std::vector<Step> current;
        current.push_back(Step::deferred_fusion(
            *output, nullptr,
            [](std::span<core::runtime::FusionJob> jobs) {
                const auto& job = jobs.front();
                *job.setup = prepare_group(
                    *job.output, {job.auxiliary_outputs, job.auxiliary_count}, *job.expression, &drops
                );
                ORMI_THROW(ormi::OperationError, "Batch preparation failed.");
            },
            unexpected_fallback, std::move(expression)
        ));

        EXPECT_THROW(core::runtime::prepare(current), ormi::OperationError);
        EXPECT_EQ(drops, 0);
        EXPECT_FALSE(output->realized());
        EXPECT_THROW(current.front().run(), ormi::OperationError);
        EXPECT_EQ(drops, 1);
        EXPECT_FALSE(output->realized());
        current.clear();
        EXPECT_EQ(drops, 1);
    }
}
