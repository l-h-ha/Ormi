#include <cmath>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/plan.h>
#include <ormi/nn/modules/batch_norm.h>
#include <ormi/nn/modules/dropout.h>

namespace {
    void BM_StatefulPrepared(benchmark::State& state, bool batch_norm, bool backward) {
        using namespace ormi::core;
        const auto input =
            std::make_shared<Tensor>(Device::cpu(), Shape{state.range(0), state.range(1)}, DType::float32);
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = std::sin(static_cast<float>(i % 127));
        }
        std::shared_ptr<ormi::nn::Module> layer;
        if (batch_norm) {
            layer = std::make_shared<ormi::nn::BatchNorm>(state.range(1));
        } else {
            layer = std::make_shared<ormi::nn::Dropout>(0.5, 93);
        }
        const auto output = (*layer)(input);
        std::vector<std::shared_ptr<Tensor>> roots{output};
        if (backward) {
            auto variables = layer->parameters();
            variables.insert(variables.begin(), input);
            const auto gradients = grad(ops::sum(ops::mul(output, output)), variables);
            roots.insert(roots.end(), gradients.begin(), gradients.end());
        }
        Plan plan(roots);
        plan.run();
        for (auto _ : state) {
            plan.run();
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * input->numel());
        state.counters["steps"] = static_cast<double>(plan.steps());
    }

    BENCHMARK_CAPTURE(BM_StatefulPrepared, batch_norm_forward, true, false)
        ->Args({32, 128})
        ->Args({1024, 128})
        ->UseRealTime();
    BENCHMARK_CAPTURE(BM_StatefulPrepared, batch_norm_backward, true, true)
        ->Args({32, 128})
        ->Args({1024, 128})
        ->UseRealTime();
    BENCHMARK_CAPTURE(BM_StatefulPrepared, dropout_forward, false, false)
        ->Args({32, 128})
        ->Args({1024, 128})
        ->UseRealTime();
    BENCHMARK_CAPTURE(BM_StatefulPrepared, dropout_backward, false, true)
        ->Args({32, 128})
        ->Args({1024, 128})
        ->UseRealTime();
}
