#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/tensor.h>

namespace {
    void bmFullDouble(benchmark::State& state) {
        for (auto _ : state) {
            auto result = ormi::core::ops::full({}, 2.5, ormi::core::Device::cpu());
            benchmark::DoNotOptimize(result);
        }
    }

    void bmFullFirstRealizeDouble(benchmark::State& state) {
        const auto elements = state.range(0);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::full({elements}, 2.5, ormi::core::Device::cpu());
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmFullFirstRealizeFloat32(benchmark::State& state) {
        const auto elements = state.range(0);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::full({elements}, 2.0f, ormi::core::Device::cpu());
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmFullZeroFirstRealizeFloat32(benchmark::State& state) {
        const auto elements = state.range(0);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::full({elements}, 0.0f, ormi::core::Device::cpu());
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmZerosLikeFirstRealizeFloat32(benchmark::State& state) {
        const auto input = ormi::core::ops::full(
            {state.range(0)},
            0.0f,
            ormi::core::Device::cpu()
        );

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::zeros_like(input);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
}

BENCHMARK(bmFullDouble)->Name("graph/full_construct_double");
BENCHMARK(bmFullFirstRealizeDouble)->Name("graph/full_first_realize_double")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmFullZeroFirstRealizeFloat32)->Name("graph/full_zero_first_realize_float32")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmZerosLikeFirstRealizeFloat32)->Name("graph/zeros_like_first_realize_float32")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmFullFirstRealizeFloat32)->Name("graph/full_first_realize_float32")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
