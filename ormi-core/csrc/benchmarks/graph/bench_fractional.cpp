#include <cstdint>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/fractional.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    double forward_terms(std::int64_t lines, std::int64_t history) {
        if (history <= 1) {
            return 0.0;
        }
        return static_cast<double>(lines) *
            static_cast<double>(history) *
            static_cast<double>(history - 1) / 2.0;
    }

    double adjoint_terms(std::int64_t lines, std::int64_t history) {
        if (history <= 1) {
            return 0.0;
        }
        const auto count = static_cast<double>(history);
        return static_cast<double>(lines) * (
            count * static_cast<double>(history - 1) / 2.0 +
            static_cast<double>(history - 1)
        );
    }

    Tensor input(std::int64_t lines, std::int64_t history) {
        auto result = ormi::core::ops::full(
            {lines, history},
            0.5,
            ormi::core::Device::cpu()
        );
        result->realize();
        return result;
    }

    void bmCaputoConstruct(benchmark::State& state) {
        const auto values = input(state.range(0), state.range(1));

        for (auto _ : state) {
            const auto result =
                ormi::core::ops::caputo(values, 0.5, 0.01);
            benchmark::DoNotOptimize(result.get());
        }
    }

    void bmCaputoPlanReplay(benchmark::State& state) {
        const auto lines = state.range(0);
        const auto history = state.range(1);
        const auto values = input(lines, history);
        const auto result =
            ormi::core::ops::caputo(values, 0.5, 0.01);
        ormi::core::Plan plan({result});
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(result.get());
        }

        state.counters["l1_terms"] = benchmark::Counter(
            static_cast<double>(state.iterations()) *
                forward_terms(lines, history),
            benchmark::Counter::kIsRate
        );
    }

    void bmCaputoAdjointPlanReplay(benchmark::State& state) {
        const auto lines = state.range(0);
        const auto history = state.range(1);
        const auto values = input(lines, history);
        const auto seed = input(lines, history);
        const auto result = ormi::core::grad(
            ormi::core::ops::caputo(values, 0.5, 0.01),
            values,
            seed
        );
        ormi::core::Plan plan({result});
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(result.get());
        }

        state.counters["l1_terms"] = benchmark::Counter(
            static_cast<double>(state.iterations()) *
                adjoint_terms(lines, history),
            benchmark::Counter::kIsRate
        );
    }

    void bmCaputoL2PlanReplay(benchmark::State& state) {
        const auto lines = state.range(0);
        const auto history = state.range(1);
        const auto values = input(lines, history);
        const auto initial_derivative = input(lines, 1);
        const auto result = ormi::core::ops::caputo(
            values,
            1.5,
            0.01,
            -1,
            ormi::core::ops::reshape(initial_derivative, {lines})
        );
        ormi::core::Plan plan({result});
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(result.get());
        }

        state.counters["l2_terms"] = benchmark::Counter(
            static_cast<double>(state.iterations()) *
                forward_terms(lines, history),
            benchmark::Counter::kIsRate
        );
    }

    void bmCaputoL2AdjointPlanReplay(benchmark::State& state) {
        const auto lines = state.range(0);
        const auto history = state.range(1);
        const auto values = input(lines, history);
        const auto seed = input(lines, history);
        const auto initial_derivative = ormi::core::ops::reshape(
            input(lines, 1),
            {lines}
        );
        const auto derivative = ormi::core::ops::caputo(
            values,
            1.5,
            0.01,
            -1,
            initial_derivative
        );
        const auto result = ormi::core::grad(derivative, values, seed);
        ormi::core::Plan plan({result});
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(result.get());
        }

        state.counters["l2_terms"] = benchmark::Counter(
            static_cast<double>(state.iterations()) *
                adjoint_terms(lines, history),
            benchmark::Counter::kIsRate
        );
    }

    BENCHMARK(bmCaputoConstruct)
        ->Name("graph/caputo_construct")
        ->Args({1, 32})
        ->Args({1, 256})
        ->Args({1, 2048})
        ->Args({1, 8192})
        ->Args({32, 256});

    BENCHMARK(bmCaputoPlanReplay)
        ->Name("graph/caputo_plan_replay")
        ->UseRealTime()
        ->Args({1, 32})
        ->Args({1, 256})
        ->Args({1, 2048})
        ->Args({1, 8192})
        ->Args({32, 256});

    BENCHMARK(bmCaputoAdjointPlanReplay)
        ->Name("graph/caputo_adjoint_plan_replay")
        ->UseRealTime()
        ->Args({1, 32})
        ->Args({1, 256})
        ->Args({1, 2048})
        ->Args({1, 8192})
        ->Args({32, 256});

    BENCHMARK(bmCaputoL2PlanReplay)
        ->Name("graph/caputo_l2_plan_replay")
        ->UseRealTime()
        ->Args({1, 32})
        ->Args({1, 256})
        ->Args({1, 2048})
        ->Args({1, 8192})
        ->Args({32, 256});

    BENCHMARK(bmCaputoL2AdjointPlanReplay)
        ->Name("graph/caputo_l2_adjoint_plan_replay")
        ->UseRealTime()
        ->Args({1, 32})
        ->Args({1, 256})
        ->Args({1, 2048})
        ->Args({1, 8192})
        ->Args({32, 256});
}
