#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> vector(std::int64_t elements) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float32
        );
        auto* data = static_cast<float*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = 1.0f;
        }

        return tensor;
    }

    std::shared_ptr<ormi::core::Tensor> doubleVector(std::int64_t elements) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float64
        );
        auto* data = static_cast<double*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = 1.0;
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

    void bmNegConstruct(benchmark::State& state) {
        const auto input = vector(1);

        for (auto _ : state) {
            auto result = ormi::core::ops::neg(input);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmExpConstruct(benchmark::State& state) {
        const auto input = vector(1);

        for (auto _ : state) {
            auto result = ormi::core::ops::exp(input);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmExpAutocastConstruct(benchmark::State& state) {
        const auto input = halfVector(1);
        const ormi::core::Autocast autocast;

        for (auto _ : state) {
            auto result = ormi::core::ops::exp(input);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmNegFirstRealize(benchmark::State& state) {
        const auto input = vector(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::neg(input);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmExpFirstRealize(benchmark::State& state) {
        const auto input = vector(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::exp(input);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmNegTransposeFirstRealize(benchmark::State& state) {
        const auto side = state.range(0);
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{side, side},
            ormi::core::DType::float32
        );
        auto* data = static_cast<float*>(input->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            data[i] = 1.0f;
        }

        const auto transposed = ormi::core::ops::transpose(input, {1, 0});

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::neg(transposed);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * side * side);
    }

    void bmPointwiseChainFirstRealize(benchmark::State& state) {
        const auto input = vector(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::sin(
                ormi::core::ops::exp(
                    ormi::core::ops::add(
                        ormi::core::ops::mul(input, input),
                        input
                    )
                )
            );
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmFloat64PointwiseChainFirstRealize(benchmark::State& state) {
        const auto input = doubleVector(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::sin(
                ormi::core::ops::exp(
                    ormi::core::ops::add(
                        ormi::core::ops::mul(input, input),
                        input
                    )
                )
            );
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmFloat16PointwiseChainFirstRealize(benchmark::State& state) {
        const auto input = halfVector(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::sin(
                ormi::core::ops::exp(
                    ormi::core::ops::add(
                        ormi::core::ops::mul(input, input),
                        input
                    )
                )
            );
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
}

BENCHMARK(bmNegConstruct)->Name("graph/neg_construct");
BENCHMARK(bmExpConstruct)->Name("graph/exp_construct");
BENCHMARK(bmExpAutocastConstruct)->Name("graph/exp_autocast_construct");
BENCHMARK(bmNegFirstRealize)
    ->Name("graph/neg_first_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmExpFirstRealize)
    ->Name("graph/exp_first_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmNegTransposeFirstRealize)
    ->Name("graph/neg_transpose_first_realize")
    ->Arg(64)
    ->Arg(256);
BENCHMARK(bmPointwiseChainFirstRealize)
    ->Name("graph/pointwise_chain_first_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmFloat64PointwiseChainFirstRealize)
    ->Name("graph/float64_pointwise_chain_first_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmFloat16PointwiseChainFirstRealize)
    ->Name("graph/float16_pointwise_chain_first_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
