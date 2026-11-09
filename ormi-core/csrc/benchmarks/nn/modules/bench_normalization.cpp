#include <cmath>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/nn/modules/normalization.h>

namespace {
    void BM_LayerNormPrepared(benchmark::State& state, bool backward) {
        using namespace ormi::core;

        const auto input =
            std::make_shared<Tensor>(Device::cpu(), Shape{state.range(0), state.range(1)}, DType::float32);
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = std::sin(static_cast<float>(i % 127));
        }
        const ormi::nn::LayerNorm layer({state.range(1)});
        const auto output = layer(input);
        std::vector<std::shared_ptr<Tensor>> roots{output};
        if (backward) {
            const auto direction = std::make_shared<Tensor>(Device::cpu(), input->shape(), DType::float32);
            auto* upstream = static_cast<float*>(direction->data());
            for (std::size_t i = 0; i < direction->numel(); ++i) {
                upstream[i] = std::cos(static_cast<float>(i % 113));
            }
            const auto objective = ops::sum(ops::mul(output, direction));
            auto variables = layer.parameters();
            variables.insert(variables.begin(), input);
            const auto gradients = grad(objective, variables);
            roots.insert(roots.end(), gradients.begin(), gradients.end());
        }

        Plan plan(roots);
        plan.run();
        for (auto _ : state) {
            plan.run();
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * state.range(0) * state.range(1));
        state.counters["steps"] = static_cast<double>(plan.steps());
    }

    BENCHMARK_CAPTURE(BM_LayerNormPrepared, forward, false)
        ->Args({1, 32})
        ->Args({32, 128})
        ->Args({1024, 128})
        ->UseRealTime();
    BENCHMARK_CAPTURE(BM_LayerNormPrepared, backward, true)
        ->Args({1, 32})
        ->Args({32, 128})
        ->Args({1024, 128})
        ->UseRealTime();
}
