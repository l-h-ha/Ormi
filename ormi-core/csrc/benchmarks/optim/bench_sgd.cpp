#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/optimizer.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> vector(std::int64_t elements, float value) {
        auto tensor = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float32
        );
        auto* data = static_cast<float*>(tensor->data());

        for (std::size_t i = 0; i < tensor->numel(); ++i) {
            data[i] = value;
        }

        return tensor;
    }

    void bmSGDConstruct(benchmark::State& state) {
        const auto parameter = vector(1, 1.0f);
        const auto gradient = vector(1, 0.25f);

        for (auto _ : state) {
            auto next = ormi::core::optim::sgd(parameter, gradient, 0.001);
            benchmark::DoNotOptimize(next);
        }
    }

    void bmSGDScaledConstruct(benchmark::State& state) {
        const auto parameter = vector(1, 1.0f);
        const auto gradient = vector(1, 256.0f);

        for (auto _ : state) {
            auto next = ormi::core::optim::sgd(parameter, gradient, 0.001, 1024.0);
            benchmark::DoNotOptimize(next);
        }
    }

    void bmSGDFirstRealize(benchmark::State& state) {
        const auto parameter = vector(state.range(0), 1.0f);
        const auto gradient = vector(state.range(0), 0.25f);

        for (auto _ : state) {
            state.PauseTiming();
            auto next = ormi::core::optim::sgd(parameter, gradient, 0.001);
            state.ResumeTiming();

            next->realize();
            benchmark::DoNotOptimize(next->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmSGDScaledFirstRealize(benchmark::State& state) {
        const auto parameter = vector(state.range(0), 1.0f);
        const auto gradient = vector(state.range(0), 256.0f);

        for (auto _ : state) {
            state.PauseTiming();
            auto next = ormi::core::optim::sgd(parameter, gradient, 0.001, 1024.0);
            state.ResumeTiming();

            next->realize();
            benchmark::DoNotOptimize(next->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmSGDOutOfPlaceKernel(benchmark::State& state) {
        const auto parameter = vector(state.range(0), 1.0f);
        const auto gradient = vector(state.range(0), 0.25f);
        auto out = vector(state.range(0), 0.0f);

        for (auto _ : state) {
            ormi::backends::cpu::sgd(*out, *parameter, *gradient, 0.000001);
            benchmark::DoNotOptimize(out->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }

    void bmSGDInPlaceKernel(benchmark::State& state) {
        auto parameter = vector(state.range(0), 1.0f);
        const auto gradient = vector(state.range(0), 0.25f);

        for (auto _ : state) {
            ormi::backends::cpu::sgd(*parameter, *parameter, *gradient, 0.000001);
            benchmark::DoNotOptimize(parameter->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
}

BENCHMARK(bmSGDConstruct)->Name("optim/sgd_construct");
BENCHMARK(bmSGDScaledConstruct)->Name("optim/sgd_scaled_construct");
BENCHMARK(bmSGDFirstRealize)
    ->Name("optim/sgd_first_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmSGDScaledFirstRealize)
    ->Name("optim/sgd_scaled_first_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmSGDOutOfPlaceKernel)
    ->Name("optim/sgd_out_of_place_kernel")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmSGDInPlaceKernel)
    ->Name("optim/sgd_in_place_kernel")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
