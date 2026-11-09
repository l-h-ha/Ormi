#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/nn/activation.h>
#include <ormi/nn/losses/classification.h>
#include <ormi/nn/losses/regression.h>

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;
    using Activation = Tensor (*)(const Tensor&);

    void BM_ActivationPrepared(benchmark::State& state, Activation activation) {
        const auto input = ormi::core::ops::full({state.range(0)}, 0.25f, ormi::core::Device::cpu());
        input->realize();
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 32.0f;
        }
        const auto output = activation(input);
        ormi::core::Plan plan({output});
        plan.run();
        for (auto _ : state) {
            plan.run();
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0));
        state.counters["steps"] = static_cast<double>(plan.steps());
    }

    void BM_ClassificationPrepared(benchmark::State& state) {
        const auto input = ormi::core::ops::full({state.range(0), 32}, 0.25f, ormi::core::Device::cpu());
        const auto target =
            ormi::core::ops::full({state.range(0)}, std::int32_t{7}, ormi::core::Device::cpu());
        input->realize();
        target->realize();
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 32.0f;
        }
        const auto loss = ormi::nn::cross_entropy(input, target);
        const auto gradient = ormi::core::grad(loss, input);
        ormi::core::Plan plan({loss, gradient});
        plan.run();
        for (auto _ : state) {
            plan.run();
            benchmark::ClobberMemory();
        }
        state.counters["steps"] = static_cast<double>(plan.steps());
    }

    BENCHMARK_CAPTURE(BM_ActivationPrepared, relu, ormi::nn::relu)
        ->Arg(16)
        ->Arg(1024)
        ->Arg(65536)
        ->UseRealTime();
    BENCHMARK_CAPTURE(BM_ActivationPrepared, gelu, ormi::nn::gelu)
        ->Arg(16)
        ->Arg(1024)
        ->Arg(65536)
        ->UseRealTime();
    BENCHMARK_CAPTURE(BM_ActivationPrepared, silu, ormi::nn::silu)
        ->Arg(16)
        ->Arg(1024)
        ->Arg(65536)
        ->UseRealTime();
    BENCHMARK(BM_ClassificationPrepared)->Arg(1)->Arg(32)->Arg(1024)->UseRealTime();
}
