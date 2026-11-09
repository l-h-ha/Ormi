#include <cstddef>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>

#include "core/runtime/dispatch/device.h"

namespace {
    using namespace ormi::core;

    std::shared_ptr<Tensor> points(std::int64_t count, Device device) {
        auto input = std::make_shared<Tensor>(device, Shape{count, 1}, DType::float64);
        std::vector<double> values(input->numel());
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.1 + 0.01 * static_cast<double>(i);
        }
        memory::copy(input->data(), device, values.data(), Device::cpu(), input->bytes());
        return input;
    }

    void bmPartialConstruct(benchmark::State& state, const char* backend) {
        const auto input = points(state.range(0), Device::from(backend));
        const auto centered = ops::sub(input, ops::mean(input, {0}, true));
        const auto output = ops::mul(ops::mul(centered, centered), centered);
        const auto order = static_cast<std::size_t>(state.range(1));
        for (auto _ : state) {
            auto result = partial(output, input, {}, {}, order);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmPartialPrepared(benchmark::State& state, const char* backend) {
        const auto device = Device::from(backend);
        const auto input = points(state.range(0), device);
        const auto centered = ops::sub(input, ops::mean(input, {0}, true));
        const auto output = ops::mul(ops::mul(centered, centered), centered);
        auto result = partial(output, input, {}, {}, static_cast<std::size_t>(state.range(1)));
        Plan plan({result});
        plan.run();
        runtime::device::synchronize(device);
        state.counters["steps"] = static_cast<double>(plan.steps());
        state.counters["storage_bytes"] = static_cast<double>(plan.storage_bytes());
        for (auto _ : state) {
            plan.run();
            runtime::device::synchronize(device);
            benchmark::DoNotOptimize(result);
        }
    }

#ifdef ORMI_ENABLE_CUDA
    void bmPartialReplay(benchmark::State& state) {
        const auto device = Device::from("cuda");
        const auto input = points(state.range(0), device);
        const auto centered = ops::sub(input, ops::mean(input, {0}, true));
        const auto output = ops::mul(ops::mul(centered, centered), centered);
        const auto result = partial(output, input, {}, {}, static_cast<std::size_t>(state.range(1)));
        Plan plan({result});
        plan.run();
        auto replay = plan.record();
        replay.run();
        runtime::device::synchronize(device);
        state.counters["steps"] = static_cast<double>(plan.steps());
        state.counters["storage_bytes"] = static_cast<double>(plan.storage_bytes());
        for (auto _ : state) {
            replay.run();
            runtime::device::synchronize(device);
        }
    }
#endif
}

BENCHMARK_CAPTURE(bmPartialConstruct, cpu, "cpu")
    ->Name("graph/partial_construct")
    ->Args({1, 1})
    ->Args({4, 1})
    ->Args({16, 1})
    ->Args({4, 2})
    ->Args({16, 2});
BENCHMARK_CAPTURE(bmPartialPrepared, cpu, "cpu")
    ->Name("graph/partial_prepared")
    ->Args({1, 1})
    ->Args({4, 1})
    ->Args({16, 1})
    ->Args({4, 2})
    ->Args({16, 2})
    ->UseRealTime();

#ifdef ORMI_ENABLE_CUDA
BENCHMARK_CAPTURE(bmPartialConstruct, cuda, "cuda")
    ->Name("cuda/partial_construct")
    ->Args({1, 1})
    ->Args({4, 1})
    ->Args({16, 1})
    ->Args({4, 2})
    ->Args({16, 2});
BENCHMARK_CAPTURE(bmPartialPrepared, cuda, "cuda")
    ->Name("cuda/partial_prepared")
    ->Args({1, 1})
    ->Args({4, 1})
    ->Args({16, 1})
    ->Args({4, 2})
    ->Args({16, 2})
    ->UseRealTime();
BENCHMARK(bmPartialReplay)
    ->Name("cuda/partial_replay")
    ->Args({1, 1})
    ->Args({4, 1})
    ->Args({16, 1})
    ->Args({4, 2})
    ->Args({16, 2})
    ->UseRealTime();
#endif
