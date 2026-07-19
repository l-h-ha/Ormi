#include <cstdint>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> inputTensor() {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3},
            ormi::core::DType::float32
        );
    }

    void bmReshapeConstruct(benchmark::State& state) {
        const auto input = inputTensor();

        for (auto _ : state) {
            auto result = ormi::core::ops::reshape(input, ormi::core::Shape{3, 2});
            benchmark::DoNotOptimize(result);
        }
    }

    void bmTransposeConstruct(benchmark::State& state) {
        const auto input = inputTensor();

        for (auto _ : state) {
            auto result = ormi::core::ops::transpose(
                input,
                std::vector<std::int64_t>{1, 0}
            );
            benchmark::DoNotOptimize(result);
        }
    }

    void bmAliasFirstRealize(benchmark::State& state) {
        const auto input = inputTensor();

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::reshape(input, ormi::core::Shape{3, 2});
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }
    }
}

BENCHMARK(bmReshapeConstruct)->Name("graph/reshape_construct");
BENCHMARK(bmTransposeConstruct)->Name("graph/transpose_construct");
BENCHMARK(bmAliasFirstRealize)->Name("graph/alias_first_realize");
