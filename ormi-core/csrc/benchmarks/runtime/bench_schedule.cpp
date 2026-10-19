#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/group.h"
#include "core/runtime/lower.h"
#include "core/runtime/schedule.h"
#include "core/runtime/step.h"

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    Tensor input() {
        auto result = ormi::core::ops::full(
            {1},
            0.5f,
            ormi::core::Device::cpu()
        );
        result->realize();
        return result;
    }

    Tensor graph(const Tensor& source, std::size_t depth) {
        auto value = source;
        for (std::size_t index = 0; index < depth; ++index) {
            value = ormi::core::ops::add(
                ormi::core::ops::mul(value, source),
                source
            );
        }
        return value;
    }

    std::vector<Tensor> aliases(const Tensor& source, std::size_t depth) {
        std::vector<Tensor> result;
        result.reserve(depth + 1);
        result.push_back(source);
        for (std::size_t index = 0; index < depth; ++index) {
            result.push_back(ormi::core::ops::reshape(result.back(), {1}));
        }
        return result;
    }

    std::vector<ormi::core::runtime::Step> build_schedule(std::span<ormi::core::Tensor* const> roots) {
        const auto candidates = ormi::core::runtime::fusion::find_candidates(roots);
        std::unordered_set<const ormi::core::Tensor*> completed;
        std::vector<ormi::core::runtime::Step> steps;
        for (auto* root : roots) {
            ormi::core::runtime::schedule::append(*root, completed, steps, candidates);
        }
        return steps;
    }

    void bmLowerRoot(benchmark::State& state) {
        const auto source = input();
        const auto root = graph(
            source,
            static_cast<std::size_t>(state.range(0))
        );

        for (auto _ : state) {
            auto step = ormi::core::runtime::lower(*root, ormi::core::runtime::Preparation::direct);
            benchmark::DoNotOptimize(step.input_count());
        }
    }

    void bmBuildSchedule(benchmark::State& state) {
        const auto source = input();
        const auto root = graph(
            source,
            static_cast<std::size_t>(state.range(0))
        );
        const std::array<ormi::core::Tensor*, 1> roots{root.get()};
        auto warmup = build_schedule(roots);
        benchmark::DoNotOptimize(warmup.size());

        for (auto _ : state) {
            auto steps = build_schedule(roots);
            benchmark::DoNotOptimize(steps.size());
        }
    }

    void bmBuildGroupedSchedule(benchmark::State& state) {
        const auto source = input();
        const auto mapped = graph(source, static_cast<std::size_t>(state.range(0)));
        std::vector<Tensor> outputs;
        std::vector<ormi::core::Tensor*> roots;
        outputs.reserve(static_cast<std::size_t>(state.range(1)));
        roots.reserve(static_cast<std::size_t>(state.range(1)));
        for (std::int64_t index = 0; index < state.range(1); ++index) {
            outputs.push_back(ormi::core::ops::neg(mapped));
            roots.push_back(outputs.back().get());
        }
        const auto warmup = build_schedule(roots);
        if (warmup.size() != 1 || warmup.front().output_count() != roots.size()) {
            state.SkipWithError("Expected one grouped fusion step.");
            return;
        }

        for (auto _ : state) {
            auto steps = build_schedule(roots);
            benchmark::DoNotOptimize(steps.size());
        }
        state.counters["outputs"] = static_cast<double>(roots.size());
    }

    void bmRunFused(benchmark::State& state) {
        const auto source = input();
        const auto depth = static_cast<std::size_t>(state.range(0));
        auto warmup = graph(source, depth);
        warmup->realize();
        warmup.reset();

        for (auto _ : state) {
            state.PauseTiming();
            auto root = graph(source, depth);
            state.ResumeTiming();

            root->realize();
            benchmark::DoNotOptimize(root->data());

            state.PauseTiming();
            root.reset();
            state.ResumeTiming();
        }
    }

    void bmRealizeAliases(benchmark::State& state) {
        const auto source = input();
        const auto depth = static_cast<std::size_t>(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto tensors = aliases(source, depth);
            state.ResumeTiming();

            tensors.back()->realize();
            benchmark::DoNotOptimize(tensors.back()->data());

            state.PauseTiming();
            tensors.clear();
            state.ResumeTiming();
        }
    }
}

#define ORMI_SCHEDULE_DEPTHS(BENCHMARK_NAME) \
    BENCHMARK_NAME->Arg(1)->Arg(4)->Arg(16)->Arg(64)

ORMI_SCHEDULE_DEPTHS(
    BENCHMARK(bmLowerRoot)
        ->Name("runtime/schedule/lower_root")
);

ORMI_SCHEDULE_DEPTHS(
    BENCHMARK(bmBuildSchedule)
        ->Name("runtime/schedule/build")
);

BENCHMARK(bmBuildGroupedSchedule)
    ->Name("runtime/schedule/build_grouped")
    ->ArgsProduct({{1, 4, 16, 64}, {2, 4, 8}})
    ->UseRealTime();

ORMI_SCHEDULE_DEPTHS(
    BENCHMARK(bmRunFused)
        ->Name("runtime/schedule/run_fused")
);

ORMI_SCHEDULE_DEPTHS(
    BENCHMARK(bmRealizeAliases)
        ->Name("runtime/schedule/realize_aliases")
);

#undef ORMI_SCHEDULE_DEPTHS
