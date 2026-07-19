#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> scalarTensor() {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float32
        );
        *static_cast<float*>(tensor->data()) = 1.0f;
        return tensor;
    }

    std::shared_ptr<ormi::core::Tensor> halfScalarTensor() {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float16
        );
        *static_cast<ormi::core::Float16*>(tensor->data()) = ormi::core::Float16(1.0f);
        return tensor;
    }

    void bmAddConstruct(benchmark::State& state) {
        const auto lhs = scalarTensor();
        const auto rhs = scalarTensor();

        for (auto _ : state) {
            auto result = ormi::core::ops::add(lhs, rhs);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmAddPromotedConstruct(benchmark::State& state) {
        const auto lhs = halfScalarTensor();
        const auto rhs = scalarTensor();

        for (auto _ : state) {
            auto result = ormi::core::ops::add(lhs, rhs);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmAddFirstRealize(benchmark::State& state) {
        const auto lhs = scalarTensor();
        const auto rhs = scalarTensor();

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::add(lhs, rhs);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }
    }

    void bmAddCachedRealize(benchmark::State& state) {
        const auto lhs = scalarTensor();
        const auto rhs = scalarTensor();
        const auto result = ormi::core::ops::add(lhs, rhs);
        result->realize();

        for (auto _ : state) {
            result->realize();
            benchmark::DoNotOptimize(result.get());
        }
    }

    void bmAddChainConstruct(benchmark::State& state) {
        const auto leaf = scalarTensor();

        for (auto _ : state) {
            auto result = leaf;

            for (std::int64_t depth = 0; depth < state.range(0); ++depth) {
                result = ormi::core::ops::add(result, leaf);
            }

            benchmark::DoNotOptimize(result.get());
        }
    }
}

BENCHMARK(bmAddConstruct)->Name("graph/add_construct");
BENCHMARK(bmAddPromotedConstruct)->Name("graph/add_promoted_construct");
BENCHMARK(bmAddFirstRealize)->Name("graph/add_first_realize");
BENCHMARK(bmAddCachedRealize)->Name("graph/add_cached_realize");
BENCHMARK(bmAddChainConstruct)
    ->Name("graph/add_chain_construct")
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000);
