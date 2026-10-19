#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;
    using Plan = ormi::core::Plan;

    Tensor input(std::int64_t elements) {
        auto result = ormi::core::ops::full(
            {elements},
            0.5f,
            ormi::core::Device::cpu()
        );
        result->realize();
        return result;
    }

    Tensor chain(const Tensor& input) {
        return ormi::core::ops::sin(
            ormi::core::ops::add(
                ormi::core::ops::mul(input, input),
                input
            )
        );
    }

    std::vector<Tensor> alias_chain(std::size_t depth) {
        std::vector<Tensor> tensors;
        tensors.reserve(depth + 1);
        tensors.push_back(input(1));

        for (std::size_t index = 0; index < depth; ++index) {
            tensors.push_back(
                ormi::core::ops::reshape(tensors.back(), {1})
            );
        }
        return tensors;
    }

    void release(std::vector<Tensor>& tensors) {
        for (auto tensor = tensors.rbegin(); tensor != tensors.rend(); ++tensor) {
            tensor->reset();
        }
    }

    void warm(const Tensor& input) {
        chain(input)->realize();
    }

    void bmPlanPrepare(benchmark::State& state) {
        const auto values = input(state.range(0));
        warm(values);

        for (auto _ : state) {
            state.PauseTiming();
            auto output = chain(values);
            std::optional<Plan> plan;
            state.ResumeTiming();

            plan.emplace(std::vector<Tensor>{output});
            benchmark::DoNotOptimize(plan->steps());

            state.PauseTiming();
            plan.reset();
            output.reset();
            state.ResumeTiming();
        }
    }

    void bmPlanFirstRun(benchmark::State& state) {
        const auto values = input(state.range(0));
        warm(values);

        for (auto _ : state) {
            state.PauseTiming();
            auto output = chain(values);
            auto plan = std::make_unique<Plan>(
                std::vector<Tensor>{output}
            );
            state.ResumeTiming();

            plan->run();
            benchmark::DoNotOptimize(output.get());

            state.PauseTiming();
            plan.reset();
            output.reset();
            state.ResumeTiming();
        }

        state.SetItemsProcessed(
            state.iterations() * state.range(0)
        );
    }

    void bmDeepPlanPrepare(benchmark::State& state) {
        auto tensors = alias_chain(
            static_cast<std::size_t>(state.range(0))
        );

        for (auto _ : state) {
            const Plan plan({tensors.back()});
            benchmark::DoNotOptimize(plan.steps());
        }

        release(tensors);
    }

    void bmPlanReplay(benchmark::State& state) {
        const auto values = input(state.range(0));
        const auto output = chain(values);
        Plan plan({output});
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(output.get());
        }

        state.SetItemsProcessed(
            state.iterations() * state.range(0)
        );
    }

    void bmPartialReductionPlanReplay(benchmark::State& state) {
        const auto values = input(state.range(0));
        const auto output = ormi::core::ops::sum(chain(values));
        Plan plan({output});
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(output.get());
        }

        state.SetItemsProcessed(
            state.iterations() * state.range(0)
        );
        state.counters["storage_buffers"] =
            static_cast<double>(plan.storage_buffers());
        state.counters["storage_bytes"] =
            static_cast<double>(plan.storage_bytes());
    }

    void bmGroupedPlanReplay(benchmark::State& state) {
        const auto values = input(state.range(0));
        const auto mapped = ormi::core::ops::add(values, values);
        std::vector<Tensor> outputs;
        outputs.reserve(static_cast<std::size_t>(state.range(1)));
        for (std::int64_t index = 0; index < state.range(1); ++index) {
            outputs.push_back(ormi::core::ops::neg(mapped));
        }
        Plan plan(outputs);
        if (plan.steps() != 1u) {
            state.SkipWithError("Expected one fused step for sibling outputs.");
            return;
        }
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(outputs.front().get());
        }

        state.SetItemsProcessed(
            state.iterations() * state.range(0) * state.range(1)
        );
        state.counters["outputs"] = static_cast<double>(outputs.size());
        state.counters["steps"] = static_cast<double>(plan.steps());
    }

    void bmRealize(benchmark::State& state) {
        const auto values = input(state.range(0));
        warm(values);

        for (auto _ : state) {
            state.PauseTiming();
            auto output = chain(values);
            state.ResumeTiming();

            output->realize();
            benchmark::DoNotOptimize(output.get());

            state.PauseTiming();
            output.reset();
            state.ResumeTiming();
        }

        state.SetItemsProcessed(
            state.iterations() * state.range(0)
        );
    }

    void bmDerivativePlanReplay(benchmark::State& state) {
        const auto x = ormi::core::ops::full(
            {},
            0.25f,
            ormi::core::Device::cpu()
        );
        const auto first_weight = ormi::core::ops::full(
            {1, 16},
            0.125f,
            ormi::core::Device::cpu()
        );
        const auto second_weight = ormi::core::ops::full(
            {16, 1},
            0.125f,
            ormi::core::Device::cpu()
        );
        x->realize();
        first_weight->realize();
        second_weight->realize();

        const auto hidden = ormi::core::ops::sin(
            ormi::core::ops::matmul(
                ormi::core::ops::reshape(x, {1, 1}),
                first_weight
            )
        );
        const auto prediction =
            ormi::core::ops::matmul(hidden, second_weight);
        const auto loss = ormi::core::ops::sum(prediction);
        const auto derivative = ormi::core::diff(
            loss,
            x,
            static_cast<std::size_t>(state.range(0))
        );
        Plan plan({derivative});
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(derivative.get());
        }

        state.counters["steps"] =
            static_cast<double>(plan.steps());
        state.counters["storage_buffers"] =
            static_cast<double>(plan.storage_buffers());
        state.counters["storage_bytes"] =
            static_cast<double>(plan.storage_bytes());
    }
}

#define ORMI_PLAN_SIZES(BENCHMARK_NAME) \
    BENCHMARK_NAME->Arg(1)->Arg(1024)->Arg(1048576)->UseRealTime()

ORMI_PLAN_SIZES(
    BENCHMARK(bmPlanPrepare)
        ->Name("runtime/plan/prepare")
);
ORMI_PLAN_SIZES(
    BENCHMARK(bmPlanFirstRun)
        ->Name("runtime/plan/first_run")
);
BENCHMARK(bmDeepPlanPrepare)
    ->Name("runtime/plan/deep_prepare")
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096);
ORMI_PLAN_SIZES(
    BENCHMARK(bmPlanReplay)
        ->Name("runtime/plan/replay")
);
BENCHMARK(bmPartialReductionPlanReplay)
    ->Name("runtime/plan/partial_reduction_replay")
    ->Arg(1048576)
    ->Arg(4194304)
    ->UseRealTime();
BENCHMARK(bmGroupedPlanReplay)
    ->Name("runtime/plan/grouped_replay")
    ->ArgsProduct({{1, 1024, 1048576}, {1, 2, 4, 8}})
    ->UseRealTime();
ORMI_PLAN_SIZES(
    BENCHMARK(bmRealize)
        ->Name("runtime/plan/realize")
);

BENCHMARK(bmDerivativePlanReplay)
    ->Name("runtime/plan/derivative_replay")
    ->Arg(1)
    ->Arg(3)
    ->Arg(5)
    ->UseRealTime();

#undef ORMI_PLAN_SIZES
