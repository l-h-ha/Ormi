#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/op.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;
    using Plan = ormi::core::Plan;
    using Update = ormi::core::Plan::Update;

    struct SetupState {
        int* runs;
        int* drops;
    };

    int setup_creates = 0;
    int setup_runs = 0;
    int setup_drops = 0;
    int fallback_runs = 0;
    std::size_t reported_buffers = 0;
    std::size_t reported_bytes = 0;

    void run_setup(void* state, ormi::core::Tensor& out) {
        auto& current = *static_cast<SetupState*>(state);
        ++*current.runs;
        *static_cast<float*>(out.data()) =
            2.0f * *static_cast<const float*>(out.parents()[0]->data());
    }

    void drop_setup(void* state) noexcept {
        auto* current = static_cast<SetupState*>(state);
        ++*current->drops;
        delete current;
    }

    ormi::core::Setup prepare_setup(ormi::core::Tensor&) {
        ++setup_creates;
        return ormi::core::Setup(
            new SetupState{&setup_runs, &setup_drops},
            run_setup,
            drop_setup,
            2,
            96
        );
    }

    void run_fallback(ormi::core::Tensor&) {
        ++fallback_runs;
    }

    const ormi::core::Op setup_op{
        .name = "setup_test",
        .output = ormi::core::Output::allocate,
        .parents = 1,
        .compute = run_fallback,
        .prepare = prepare_setup,
        .adjoint = nullptr,
        .scalar = std::monostate{}
    };

    ormi::core::Setup prepare_reported_storage(ormi::core::Tensor&) {
        ++setup_creates;
        return ormi::core::Setup(
            new SetupState{&setup_runs, &setup_drops},
            [](void* state, ormi::core::Tensor&) {
                ++*static_cast<SetupState*>(state)->runs;
            },
            drop_setup,
            reported_buffers,
            reported_bytes
        );
    }

    const ormi::core::Op storage_op{
        .name = "reported_storage_test",
        .output = ormi::core::Output::allocate,
        .parents = 1,
        .compute = run_fallback,
        .prepare = prepare_reported_storage,
        .adjoint = nullptr,
        .scalar = std::monostate{}
    };

    Tensor scalar(float value) {
        auto result = ormi::core::ops::full(
            {},
            value,
            ormi::core::Device::cpu()
        );
        result->realize();
        return result;
    }

    Tensor setup_tensor(const Tensor& input) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<setup_op>(),
            std::vector<Tensor>{input},
            input->device(),
            input->shape(),
            input->dtype()
        );
    }

    Tensor storage_tensor(const Tensor& input, ormi::core::Shape shape) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::op_ref<storage_op>(), std::vector<Tensor>{input},
            input->device(), std::move(shape), input->dtype()
        );
    }

    TEST(CPUPlanTest, reportsNoStorageForAliasOnlyPlan) {
        const auto input = ormi::core::ops::full({2, 3}, 1.0f, ormi::core::Device::cpu());
        input->realize();
        const auto output = ormi::core::ops::transpose(input, {1, 0});
        Plan plan({output});

        EXPECT_EQ(plan.steps(), 1u);
        EXPECT_EQ(plan.storage_buffers(), 0u);
        EXPECT_EQ(plan.storage_bytes(), 0u);
        plan.run();
        EXPECT_EQ(output->buffer(), input->buffer());
        EXPECT_EQ(plan.storage_buffers(), 0u);
        EXPECT_EQ(plan.storage_bytes(), 0u);
    }

    TEST(CPUPlanTest, skipsEmptyOutputAndCountsPreparedWorkspace) {
        setup_creates = setup_runs = setup_drops = 0;
        reported_buffers = 2;
        reported_bytes = 96;
        const auto output = storage_tensor(scalar(1.0f), {0, 7});
        {
            Plan plan({output});
            EXPECT_EQ(setup_creates, 1);
            EXPECT_EQ(setup_runs, 0);
            EXPECT_EQ(plan.storage_buffers(), 2u);
            EXPECT_EQ(plan.storage_bytes(), 96u);
            EXPECT_FALSE(output->realized());
            plan.run();
            EXPECT_TRUE(output->realized());
            EXPECT_EQ(setup_runs, 1);
            EXPECT_EQ(plan.storage_buffers(), 2u);
            EXPECT_EQ(plan.storage_bytes(), 96u);
        }
        EXPECT_EQ(setup_drops, 1);
    }

    TEST(CPUPlanTest, acceptsMaximumReportedStorageForEmptyOutput) {
        setup_creates = setup_runs = setup_drops = 0;
        reported_buffers = std::numeric_limits<std::size_t>::max();
        reported_bytes = std::numeric_limits<std::size_t>::max();
        {
            Plan plan({storage_tensor(scalar(1.0f), {0})});
            EXPECT_EQ(plan.storage_buffers(), reported_buffers);
            EXPECT_EQ(plan.storage_bytes(), reported_bytes);
            EXPECT_EQ(setup_creates, 1);
            EXPECT_EQ(setup_runs, 0);
        }
        EXPECT_EQ(setup_drops, 1);
    }

    TEST(CPUPlanTest, rejectsAggregateStorageOverflowAndReleasesPreparedStates) {
        const auto input = scalar(1.0f);
        for (const bool buffers : {true, false}) {
            SCOPED_TRACE(buffers ? "buffer count" : "byte count");
            setup_creates = setup_runs = setup_drops = 0;
            const auto count = std::numeric_limits<std::size_t>::max() / 2 + 1;
            reported_buffers = buffers ? count : 0;
            reported_bytes = buffers ? 0 : count;
            const auto first = storage_tensor(input, {0});
            const auto second = storage_tensor(input, {0});

            EXPECT_THROW(Plan({first, second}), ormi::MemoryError);
            EXPECT_EQ(setup_creates, 2);
            EXPECT_EQ(setup_drops, 2);
            EXPECT_EQ(setup_runs, 0);
            EXPECT_FALSE(first->realized());
            EXPECT_FALSE(second->realized());
        }
    }

    TEST(CPUPlanTest, rejectsOutputAndWorkspaceStorageOverflow) {
        const auto input = scalar(1.0f);
        for (const bool buffers : {true, false}) {
            SCOPED_TRACE(buffers ? "buffer count" : "byte count");
            setup_creates = setup_runs = setup_drops = 0;
            reported_buffers = buffers ? std::numeric_limits<std::size_t>::max() : 0;
            reported_bytes = buffers ? 0 : std::numeric_limits<std::size_t>::max();
            const auto output = storage_tensor(input, {});

            EXPECT_THROW(Plan({output}), ormi::MemoryError);
            EXPECT_EQ(setup_creates, 1);
            EXPECT_EQ(setup_drops, 1);
            EXPECT_EQ(setup_runs, 0);
            EXPECT_FALSE(output->realized());
        }
    }

    TEST(CPUPlanTest, releasesPreparedStateAfterTensorByteOverflow) {
        setup_creates = setup_runs = setup_drops = 0;
        reported_buffers = 2;
        reported_bytes = 96;
        const auto output = storage_tensor(scalar(1.0f),
            {static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float) + 1)});

        EXPECT_THROW(Plan({output}), ormi::ShapeError);
        EXPECT_EQ(setup_creates, 1);
        EXPECT_EQ(setup_drops, 1);
        EXPECT_EQ(setup_runs, 0);
        EXPECT_FALSE(output->realized());
    }

    TEST(CPUPlanTest, ownsPreparedStateAndReportsItsStorage) {
        setup_creates = 0;
        setup_runs = 0;
        setup_drops = 0;
        fallback_runs = 0;

        {
            const auto input = scalar(3.0f);
            const auto output = setup_tensor(input);
            Plan plan({output});

            EXPECT_EQ(setup_creates, 1);
            EXPECT_EQ(setup_runs, 0);
            EXPECT_EQ(setup_drops, 0);
            EXPECT_EQ(plan.storage_buffers(), 3u);
            EXPECT_EQ(plan.storage_bytes(), 96u + sizeof(float));

            plan.run();
            plan.run();
            EXPECT_EQ(setup_runs, 2);
            EXPECT_EQ(fallback_runs, 0);
            EXPECT_FLOAT_EQ(
                *static_cast<const float*>(output->data()),
                6.0f
            );
        }

        EXPECT_EQ(setup_drops, 1);
    }

    TEST(CPUPlanTest, releasesPreparedStateAfterOrdinaryRealization) {
        setup_creates = 0;
        setup_runs = 0;
        setup_drops = 0;
        fallback_runs = 0;

        const auto output = setup_tensor(scalar(4.0f));
        output->realize();

        EXPECT_EQ(setup_creates, 1);
        EXPECT_EQ(setup_runs, 1);
        EXPECT_EQ(setup_drops, 1);
        EXPECT_EQ(fallback_runs, 0);
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(output->data()),
            8.0f
        );
    }

    TEST(CPUPlanTest, replaysPreparedFusion) {
        auto input = ormi::core::ops::full(
            {3},
            0.0f,
            ormi::core::Device::cpu()
        );
        input->realize();
        auto* input_data = static_cast<float*>(input->data());
        input_data[0] = -0.5f;
        input_data[1] = 0.0f;
        input_data[2] = 0.5f;

        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(
                ormi::core::ops::mul(input, input),
                input
            )
        );
        Plan plan({output});

        EXPECT_FALSE(output->realized());
        EXPECT_EQ(plan.steps(), 1u);

        plan.run();
        const auto* first = static_cast<const float*>(output->data());
        EXPECT_FLOAT_EQ(first[0], std::sin(-0.25f));
        EXPECT_FLOAT_EQ(first[1], 0.0f);
        EXPECT_FLOAT_EQ(first[2], std::sin(0.75f));

        input_data[0] = 1.0f;
        input_data[1] = 2.0f;
        input_data[2] = 3.0f;
        plan.run();

        const auto* second = static_cast<const float*>(output->data());
        EXPECT_FLOAT_EQ(second[0], std::sin(2.0f));
        EXPECT_FLOAT_EQ(second[1], std::sin(6.0f));
        EXPECT_FLOAT_EQ(second[2], std::sin(12.0f));
    }

    TEST(CPUPlanTest, replaysPairedTrigOutputsInOneStep) {
        auto input = ormi::core::ops::full(
            {3},
            0.0f,
            ormi::core::Device::cpu()
        );
        input->realize();
        auto* values = static_cast<float*>(input->data());
        values[0] = -0.5f;
        values[1] = 0.0f;
        values[2] = 0.5f;

        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto sine = ormi::core::ops::sin(mapped);
        const auto cosine = ormi::core::ops::cos(mapped);
        Plan plan({sine, cosine});

        EXPECT_EQ(plan.steps(), 1u);
        EXPECT_EQ(plan.storage_buffers(), 2u);
        EXPECT_EQ(plan.storage_bytes(), 2u * sine->bytes());

        plan.run();
        const auto* sine_values = static_cast<const float*>(sine->data());
        const auto* cosine_values = static_cast<const float*>(cosine->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto expected = values[i] * values[i] + values[i];
            EXPECT_FLOAT_EQ(sine_values[i], std::sin(expected));
            EXPECT_FLOAT_EQ(cosine_values[i], std::cos(expected));
        }

        values[0] = 1.0f;
        values[1] = 2.0f;
        values[2] = 3.0f;
        plan.run();
        EXPECT_FLOAT_EQ(
            static_cast<const float*>(sine->data())[1],
            std::sin(6.0f)
        );
        EXPECT_FLOAT_EQ(
            static_cast<const float*>(cosine->data())[1],
            std::cos(6.0f)
        );
    }

    TEST(CPUPlanTest, replaysGenericSiblingOutputsInOneStep) {
        auto input = ormi::core::ops::full(
            {3},
            0.0f,
            ormi::core::Device::cpu()
        );
        input->realize();
        auto* values = static_cast<float*>(input->data());
        values[0] = -0.5f;
        values[1] = 0.0f;
        values[2] = 0.5f;

        const auto mapped = ormi::core::ops::add(input, input);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        Plan plan({negative, exponential});

        EXPECT_EQ(plan.steps(), 1u);
        EXPECT_EQ(plan.storage_buffers(), 2u);
        EXPECT_EQ(plan.storage_bytes(), 2u * negative->bytes());

        plan.run();
        const auto* negative_values =
            static_cast<const float*>(negative->data());
        const auto* exponential_values =
            static_cast<const float*>(exponential->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto expected = 2.0f * values[i];
            EXPECT_FLOAT_EQ(negative_values[i], -expected);
            EXPECT_FLOAT_EQ(exponential_values[i], std::exp(expected));
        }

        const auto negative_buffer = negative->buffer();
        const auto exponential_buffer = exponential->buffer();
        values[0] = 0.25f;
        values[1] = 1.0f;
        values[2] = 1.5f;
        plan.run();
        EXPECT_EQ(negative->buffer(), negative_buffer);
        EXPECT_EQ(exponential->buffer(), exponential_buffer);
        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto expected = 2.0f * values[i];
            EXPECT_FLOAT_EQ(negative_values[i], -expected);
            EXPECT_FLOAT_EQ(exponential_values[i], std::exp(expected));
        }
    }

    TEST(CPUPlanTest, replaysPairedTrigGradient) {
        auto input = ormi::core::ops::full(
            {4},
            0.0f,
            ormi::core::Device::cpu()
        );
        input->realize();
        auto* values = static_cast<float*>(input->data());
        values[0] = -0.5f;
        values[1] = 0.0f;
        values[2] = 0.5f;
        values[3] = 1.0f;

        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto loss = ormi::core::ops::sum(
            ormi::core::ops::sin(mapped)
        );
        const auto gradient = ormi::core::grad(loss, input);
        Plan plan({loss, gradient});

        plan.run();

        float expected_loss = 0.0f;
        const auto* gradient_values =
            static_cast<const float*>(gradient->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto mapped_value = values[i] * values[i] + values[i];
            expected_loss += std::sin(mapped_value);
            EXPECT_NEAR(
                gradient_values[i],
                std::cos(mapped_value) * (2.0f * values[i] + 1.0f),
                1e-6f
            );
        }
        EXPECT_NEAR(
            *static_cast<const float*>(loss->data()),
            expected_loss,
            1e-6f
        );
    }

    TEST(CPUPlanTest, ownsPreparedPartialReductionStorage) {
        constexpr std::int64_t elements = 1024 * 1024;
        const auto input = ormi::core::ops::full(
            {elements},
            0.5,
            ormi::core::Device::cpu()
        );
        input->realize();
        const auto output = ormi::core::ops::sum(
            ormi::core::ops::add(
                ormi::core::ops::mul(input, input),
                input
            )
        );
        Plan plan({output});

        EXPECT_EQ(plan.steps(), 1u);
        if (
            plan.storage_buffers() == 1u &&
            plan.storage_bytes() == sizeof(double)
        ) {
            GTEST_SKIP()
                << "This check requires a parallel partial reduction.";
        }
        EXPECT_EQ(plan.storage_buffers(), 2u);
        EXPECT_GT(plan.storage_bytes(), sizeof(double));

        const auto expected = 0.75 * static_cast<double>(elements);
        plan.run();
        const auto first = *static_cast<const double*>(output->data());
        plan.run();
        const auto second = *static_cast<const double*>(output->data());

        EXPECT_DOUBLE_EQ(first, expected);
        EXPECT_DOUBLE_EQ(second, expected);
    }

    TEST(CPUPlanTest, ownsGraphAndRemainsMovable) {
        std::weak_ptr<ormi::core::Tensor> output;
        auto plan = [&output] {
            const auto input = scalar(0.5f);
            const auto result = ormi::core::ops::sin(
                ormi::core::ops::add(
                    ormi::core::ops::mul(input, input),
                    input
                )
            );
            output = result;
            return Plan({result});
        }();

        ASSERT_FALSE(output.expired());

        Plan moved(std::move(plan));
        moved.run();

        const auto result = output.lock();
        ASSERT_NE(result, nullptr);
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(result->data()),
            std::sin(0.75f)
        );
    }

    TEST(CPUPlanTest, treatsRealizedTensorAsInputBoundary) {
        const auto input = scalar(2.0f);
        const auto boundary =
            ormi::core::ops::add(input, input);
        boundary->realize();
        const auto output =
            ormi::core::ops::mul(boundary, boundary);
        Plan plan({output});

        EXPECT_EQ(plan.steps(), 1u);
        plan.run();
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(output->data()),
            16.0f
        );

        *static_cast<float*>(boundary->data()) = 3.0f;
        plan.run();
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(output->data()),
            9.0f
        );
    }

    TEST(CPUPlanTest, replaysAliases) {
        auto input = ormi::core::ops::full(
            {2, 3},
            0.0f,
            ormi::core::Device::cpu()
        );
        input->realize();
        auto* input_data = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < 6; ++i) {
            input_data[i] = static_cast<float>(i + 1);
        }

        const auto transposed =
            ormi::core::ops::transpose(input, {1, 0});
        const auto output =
            ormi::core::ops::add(transposed, transposed);
        Plan plan({output});

        EXPECT_EQ(plan.steps(), 2u);
        EXPECT_EQ(plan.storage_buffers(), 1u);
        EXPECT_EQ(plan.storage_bytes(), output->bytes());
        plan.run();

        const auto* first = static_cast<const float*>(output->data());
        EXPECT_FLOAT_EQ(first[0], 2.0f);
        EXPECT_FLOAT_EQ(first[1], 8.0f);
        EXPECT_FLOAT_EQ(first[2], 4.0f);
        EXPECT_FLOAT_EQ(first[3], 10.0f);
        EXPECT_FLOAT_EQ(first[4], 6.0f);
        EXPECT_FLOAT_EQ(first[5], 12.0f);

        input_data[0] = 10.0f;
        plan.run();
        EXPECT_FLOAT_EQ(
            static_cast<const float*>(output->data())[0],
            20.0f
        );
    }

    TEST(CPUPlanTest, schedulesSharedDependenciesOnce) {
        const auto lhs = ormi::core::ops::full(
            {2, 2},
            1.0f,
            ormi::core::Device::cpu()
        );
        const auto rhs = ormi::core::ops::full(
            {2, 2},
            2.0f,
            ormi::core::Device::cpu()
        );
        lhs->realize();
        rhs->realize();

        const auto shared = ormi::core::ops::matmul(lhs, rhs);
        const auto first = ormi::core::ops::sin(
            ormi::core::ops::add(shared, shared)
        );
        const auto second = ormi::core::ops::cos(
            ormi::core::ops::add(shared, shared)
        );
        Plan plan({first, second});

        EXPECT_EQ(plan.steps(), 3u);
        EXPECT_EQ(plan.storage_buffers(), 3u);
        EXPECT_EQ(plan.storage_bytes(), 3u * 4u * sizeof(float));
        plan.run();
        EXPECT_FLOAT_EQ(
            static_cast<const float*>(first->data())[0],
            std::sin(8.0f)
        );
        EXPECT_FLOAT_EQ(
            static_cast<const float*>(second->data())[0],
            std::cos(8.0f)
        );
    }

    TEST(CPUPlanTest, appliesUpdatesAfterEveryRun) {
        const auto parameter = scalar(2.0f);
        const auto gradient = scalar(1.0f);
        const auto next =
            ormi::core::optim::sgd(parameter, gradient, 0.1);
        Plan plan(
            {},
            std::vector<Update>{{parameter, next}}
        );

        EXPECT_EQ(plan.steps(), 1u);
        plan.run();
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(parameter->data()),
            1.9f
        );

        plan.run();
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(parameter->data()),
            1.8f
        );
    }

    TEST(CPUPlanTest, evaluatesAllUpdatesBeforeApplyingAny) {
        const auto first = scalar(2.0f);
        const auto second = scalar(3.0f);
        const auto next_first =
            ormi::core::ops::add(first, second);
        const auto next_second =
            ormi::core::ops::mul(first, second);
        Plan plan(
            {},
            std::vector<Update>{
                {first, next_first},
                {second, next_second}
            }
        );

        plan.run();
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(first->data()),
            5.0f
        );
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(second->data()),
            6.0f
        );

        plan.run();
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(first->data()),
            11.0f
        );
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(second->data()),
            30.0f
        );
    }

    TEST(CPUPlanTest, replaysLazyGradientAndOptimizer) {
        const auto parameter = scalar(2.0f);
        const auto loss =
            ormi::core::ops::mul(parameter, parameter);
        const auto gradient =
            ormi::core::grad(loss, parameter);
        const auto next =
            ormi::core::optim::sgd(parameter, gradient, 0.1);
        Plan plan(
            {loss},
            std::vector<Update>{{parameter, next}}
        );

        plan.run();
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(loss->data()),
            4.0f
        );
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(parameter->data()),
            1.6f
        );

        plan.run();
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(loss->data()),
            2.56f
        );
        EXPECT_FLOAT_EQ(
            *static_cast<const float*>(parameter->data()),
            1.28f
        );
    }

    TEST(CPUPlanTest, rejectsUnrealizedUpdateDestination) {
        const auto destination =
            ormi::core::ops::full(
                {},
                2.0f,
                ormi::core::Device::cpu()
            );
        const auto value = scalar(1.0f);

        EXPECT_THROW(
            Plan(
                {},
                std::vector<Update>{{destination, value}}
            ),
            ormi::BufferError
        );
    }

    TEST(CPUPlanTest, rejectsNoncontiguousUpdate) {
        const auto base = ormi::core::ops::full(
            {2, 3},
            1.0f,
            ormi::core::Device::cpu()
        );
        base->realize();
        const auto destination =
            ormi::core::ops::transpose(base, {1, 0});
        destination->realize();
        const auto value = ormi::core::ops::full(
            {3, 2},
            2.0f,
            ormi::core::Device::cpu()
        );

        EXPECT_THROW(
            Plan(
                {},
                std::vector<Update>{{destination, value}}
            ),
            ormi::OperationError
        );
    }

    TEST(CPUPlanTest, rejectsOverlappingUpdateAliases) {
        const auto base = ormi::core::ops::full(
            {3},
            1.0f,
            ormi::core::Device::cpu()
        );
        base->realize();
        const auto destination = std::make_shared<ormi::core::Tensor>(
            base->buffer(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        const auto value = std::make_shared<ormi::core::Tensor>(
            base->buffer(),
            ormi::core::Shape{2},
            ormi::core::DType::float32,
            std::vector<std::int64_t>{},
            sizeof(float)
        );
        Plan plan(
            {},
            std::vector<Update>{{destination, value}}
        );

        EXPECT_THROW(plan.run(), ormi::OperationError);
    }

    TEST(CPUPlanTest, rejectsEmptyPlan) {
        EXPECT_THROW(
            Plan(std::vector<Tensor>{}),
            ormi::OperationError
        );
    }

    TEST(CPUPlanTest, rejectsRealizedOutput) {
        const auto output = scalar(1.0f);

        EXPECT_THROW(
            Plan(std::vector<Tensor>{output}),
            ormi::OperationError
        );
    }

    TEST(CPUPlanTest, rejectsDuplicateUpdateDestination) {
        const auto destination = scalar(1.0f);
        const auto first = ormi::core::ops::add(
            destination,
            scalar(1.0f)
        );
        const auto second = ormi::core::ops::mul(
            destination,
            scalar(2.0f)
        );

        EXPECT_THROW(
            Plan(
                {},
                std::vector<Update>{
                    {destination, first},
                    {destination, second}
                }
            ),
            ormi::OperationError
        );
    }

    TEST(CPUPlanTest, rejectsRecordingOnUnsupportedBackend) {
        const auto input = scalar(2.0f);
        const auto output = ormi::core::ops::mul(input, input);
        Plan plan({output});

        EXPECT_THROW(
            static_cast<void>(plan.record()),
            ormi::OperationError
        );

        plan.run();
        EXPECT_THROW(static_cast<void>(plan.record()), ormi::OperationError);
        plan.run();
        EXPECT_FLOAT_EQ(*static_cast<const float*>(output->data()), 4.0f);
    }

    TEST(CPUPlanTest, rejectsNullOutput) {
        EXPECT_THROW(
            Plan(std::vector<Tensor>{nullptr}),
            ormi::OperationError
        );
    }
}
