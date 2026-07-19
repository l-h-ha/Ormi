#include <cstdint>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> inputTensor() {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float32
        );
        auto* data = static_cast<float*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = 1.0f;
        }

        return tensor;
    }

    void bmSumConstruct(benchmark::State& state) {
        const auto input = inputTensor();

        for (auto _ : state) {
            auto result = ormi::core::ops::sum(
                input,
                std::vector<std::int64_t>{0},
                false
            );
            benchmark::DoNotOptimize(result);
        }
    }

    void bmSumFirstRealize(benchmark::State& state) {
        const auto input = inputTensor();

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::sum(input);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }
    }
}

BENCHMARK(bmSumConstruct)->Name("graph/sum_construct");
BENCHMARK(bmSumFirstRealize)->Name("graph/sum_first_realize");
