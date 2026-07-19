#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> floatVector(std::int64_t elements) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float32
        );
        auto* data = static_cast<float*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = 0.5f;
        }

        return tensor;
    }

    std::shared_ptr<ormi::core::Tensor> halfVector(std::int64_t elements) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float16
        );
        auto* data = static_cast<ormi::core::Float16*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = ormi::core::Float16(0.5f);
        }

        return tensor;
    }

    void bmCastConstruct(benchmark::State& state) {
        const auto input = floatVector(1);

        for (auto _ : state) {
            auto result = ormi::core::ops::cast(input, ormi::core::DType::float16);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmFloat32ToFloat16(benchmark::State& state) {
        const auto input = floatVector(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::cast(input, ormi::core::DType::float16);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmFloat16ToFloat32(benchmark::State& state) {
        const auto input = halfVector(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::cast(input, ormi::core::DType::float32);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
}

BENCHMARK(bmCastConstruct)->Name("graph/cast_construct");
BENCHMARK(bmFloat32ToFloat16)
    ->Name("graph/cast_float32_to_float16")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmFloat16ToFloat32)
    ->Name("graph/cast_float16_to_float32")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
