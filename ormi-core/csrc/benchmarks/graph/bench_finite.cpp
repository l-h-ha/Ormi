#include <cstdint>
#include <limits>
#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> vector(std::int64_t elements) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float32
        );
        auto* data = static_cast<float*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = 1.0f;
        }

        return tensor;
    }

    void bmAllFiniteConstruct(benchmark::State& state) {
        const auto input = vector(1);

        for (auto _ : state) {
            auto result = ormi::core::ops::all_finite(input);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmAllFiniteFirstRealize(benchmark::State& state) {
        const auto input = vector(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::all_finite(input);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmAllFiniteEarlyExit(benchmark::State& state) {
        const auto input = vector(state.range(0));
        static_cast<float*>(input->data())[0] = std::numeric_limits<float>::infinity();

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::all_finite(input);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }
    }
}

BENCHMARK(bmAllFiniteConstruct)->Name("graph/all_finite_construct");
BENCHMARK(bmAllFiniteFirstRealize)
    ->Name("graph/all_finite_first_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmAllFiniteEarlyExit)
    ->Name("graph/all_finite_early_exit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
