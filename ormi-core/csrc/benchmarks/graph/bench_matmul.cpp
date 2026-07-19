#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> matrix(std::int64_t side) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{side, side},
            ormi::core::DType::float32
        );
        auto* data = static_cast<float*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = 1.0f;
        }

        return tensor;
    }

    std::shared_ptr<ormi::core::Tensor> halfMatrix(std::int64_t side) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{side, side},
            ormi::core::DType::float16
        );
        auto* data = static_cast<ormi::core::Float16*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = ormi::core::Float16(1.0f);
        }

        return tensor;
    }

    void bmMatmulConstruct(benchmark::State& state) {
        const auto lhs = matrix(2);
        const auto rhs = matrix(2);

        for (auto _ : state) {
            auto result = ormi::core::ops::matmul(lhs, rhs);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmMatmulPromotedConstruct(benchmark::State& state) {
        const auto lhs = halfMatrix(2);
        const auto rhs = matrix(2);

        for (auto _ : state) {
            auto result = ormi::core::ops::matmul(lhs, rhs);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmMatmulAutocastConstruct(benchmark::State& state) {
        const auto lhs = matrix(2);
        const auto rhs = matrix(2);
        const ormi::core::Autocast autocast;

        for (auto _ : state) {
            auto result = ormi::core::ops::matmul(lhs, rhs);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmMatmulFirstRealize(benchmark::State& state) {
        const auto side = state.range(0);
        const auto lhs = matrix(side);
        const auto rhs = matrix(side);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::matmul(lhs, rhs);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        const auto operations =
            2.0 * static_cast<double>(side) *
            static_cast<double>(side) *
            static_cast<double>(side);
        state.counters["FLOP/s"] = benchmark::Counter(
            operations * static_cast<double>(state.iterations()),
            benchmark::Counter::kIsRate
        );
    }

    void bmMatmulAutocastFirstRealize(benchmark::State& state) {
        const auto side = state.range(0);
        const auto lhs = matrix(side);
        const auto rhs = matrix(side);
        const ormi::core::Autocast autocast;

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::matmul(lhs, rhs);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        const auto operations =
            2.0 * static_cast<double>(side) *
            static_cast<double>(side) *
            static_cast<double>(side);
        state.counters["FLOP/s"] = benchmark::Counter(
            operations * static_cast<double>(state.iterations()),
            benchmark::Counter::kIsRate
        );
    }
}

BENCHMARK(bmMatmulConstruct)->Name("graph/matmul_construct");
BENCHMARK(bmMatmulPromotedConstruct)->Name("graph/matmul_promoted_construct");
BENCHMARK(bmMatmulAutocastConstruct)->Name("graph/matmul_autocast_construct");
BENCHMARK(bmMatmulFirstRealize)
    ->Name("graph/matmul_first_realize")
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->Arg(512);
BENCHMARK(bmMatmulAutocastFirstRealize)
    ->Name("graph/matmul_autocast_first_realize")
    ->Arg(16)
    ->Arg(64);
