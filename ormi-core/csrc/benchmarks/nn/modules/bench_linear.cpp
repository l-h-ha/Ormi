#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/plan.h>
#include <ormi/nn/modules/linear.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> manual(
        const std::shared_ptr<ormi::core::Tensor>& input,
        const std::shared_ptr<ormi::core::Tensor>& weight,
        const std::shared_ptr<ormi::core::Tensor>& bias
    ) {
        return ormi::core::ops::add(ormi::core::ops::matmul(input, weight), bias);
    }

    void BM_NNLinearGraph(benchmark::State& state) {
        const ormi::nn::Linear layer(16, 32, true, ormi::core::DType::float32, ormi::core::Device::cpu(), 7);
        const auto input = ormi::core::ops::full({state.range(0), 16}, 0.25f, ormi::core::Device::cpu());
        for (auto _ : state) {
            auto output = layer(input);
            benchmark::DoNotOptimize(output);
        }
    }

    void BM_ManualLinearGraph(benchmark::State& state) {
        const ormi::nn::Linear layer(16, 32, true, ormi::core::DType::float32, ormi::core::Device::cpu(), 7);
        const auto input = ormi::core::ops::full({state.range(0), 16}, 0.25f, ormi::core::Device::cpu());
        const auto weight = layer.weight();
        const auto bias = layer.bias();
        for (auto _ : state) {
            auto output = manual(input, weight, bias);
            benchmark::DoNotOptimize(output);
        }
    }

    void BM_LinearPrepared(benchmark::State& state, bool use_layer) {
        const ormi::nn::Linear layer(16, 32, true, ormi::core::DType::float32, ormi::core::Device::cpu(), 7);
        const auto input = ormi::core::ops::full({state.range(0), 16}, 0.25f, ormi::core::Device::cpu());
        input->realize();
        const auto output = use_layer ? layer(input) : manual(input, layer.weight(), layer.bias());
        ormi::core::Plan plan({output});
        plan.run();
        for (auto _ : state) {
            plan.run();
            benchmark::ClobberMemory();
        }
        state.counters["steps"] = static_cast<double>(plan.steps());
        state.counters["storage_bytes"] = static_cast<double>(plan.storage_bytes());
    }

    BENCHMARK(BM_NNLinearGraph)->Arg(1)->Arg(32)->Arg(1024)->UseRealTime();
    BENCHMARK(BM_ManualLinearGraph)->Arg(1)->Arg(32)->Arg(1024)->UseRealTime();
    BENCHMARK_CAPTURE(BM_LinearPrepared, nn, true)->Arg(1)->Arg(32)->Arg(1024)->UseRealTime();
    BENCHMARK_CAPTURE(BM_LinearPrepared, manual, false)->Arg(1)->Arg(32)->Arg(1024)->UseRealTime();
}
