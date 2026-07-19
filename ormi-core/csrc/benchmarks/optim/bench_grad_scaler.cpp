#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/optim/grad_scaler.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

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

    ormi::core::optim::ScalePolicy stablePolicy() {
        return ormi::core::optim::ScalePolicy{
            .interval = std::numeric_limits<std::size_t>::max()
        };
    }

    void bmGradScalerScaleConstruct(benchmark::State& state) {
        const auto loss = vector(1, 2.0f);
        const ormi::core::optim::GradScaler scaler(1024.0);

        for (auto _ : state) {
            auto scaled = scaler.scale(loss);
            benchmark::DoNotOptimize(scaled);
        }
    }

    void bmGradScalerBatchStepFinite(benchmark::State& state) {
        const auto parameter_count = static_cast<std::size_t>(state.range(0));
        const auto elements = state.range(1);
        std::vector<std::shared_ptr<ormi::core::Tensor>> parameters;
        std::vector<std::shared_ptr<ormi::core::Tensor>> gradients;
        parameters.reserve(parameter_count);
        gradients.reserve(parameter_count);

        for (std::size_t i = 0; i < parameter_count; ++i) {
            parameters.push_back(vector(elements, 1.0f));
            gradients.push_back(vector(elements, 256.0f));
        }

        ormi::core::optim::GradScaler scaler(1024.0, stablePolicy());

        for (auto _ : state) {
            auto next = scaler.step(parameters, gradients, 0.001);
            benchmark::DoNotOptimize(next);
        }

        state.SetItemsProcessed(state.iterations() * parameter_count * elements);
    }

    void bmGradScalerBatchStepOverflow(benchmark::State& state) {
        const auto parameter_count = static_cast<std::size_t>(state.range(0));
        const auto elements = state.range(1);
        std::vector<std::shared_ptr<ormi::core::Tensor>> parameters;
        std::vector<std::shared_ptr<ormi::core::Tensor>> gradients;
        parameters.reserve(parameter_count);
        gradients.reserve(parameter_count);

        for (std::size_t i = 0; i < parameter_count; ++i) {
            parameters.push_back(vector(elements, 1.0f));
            gradients.push_back(vector(elements, 256.0f));
        }

        *static_cast<float*>(gradients[0]->data()) = std::numeric_limits<float>::infinity();
        ormi::core::optim::GradScaler scaler(1024.0, stablePolicy());

        for (auto _ : state) {
            auto next = scaler.step(parameters, gradients, 0.001);
            benchmark::DoNotOptimize(next);
        }
    }

    void bmGradScalerStepAndRealize(benchmark::State& state) {
        const auto parameter = vector(state.range(0), 1.0f);
        const auto gradient = vector(state.range(0), 256.0f);
        ormi::core::optim::GradScaler scaler(1024.0, stablePolicy());

        for (auto _ : state) {
            auto next = scaler.step(parameter, gradient, 0.001);
            benchmark::DoNotOptimize(next);
            (*next)->realize();
            benchmark::DoNotOptimize((*next)->data());
        }

        state.SetItemsProcessed(state.iterations() * state.range(0));
    }
}

BENCHMARK(bmGradScalerScaleConstruct)->Name("optim/grad_scaler_scale_construct");
BENCHMARK(bmGradScalerBatchStepFinite)
    ->Name("optim/grad_scaler_batch_step_finite")
    ->Args({1, 1024})
    ->Args({16, 1024})
    ->Args({1, 1048576});
BENCHMARK(bmGradScalerBatchStepOverflow)
    ->Name("optim/grad_scaler_batch_step_overflow")
    ->Args({1, 1024})
    ->Args({16, 1024})
    ->Args({1, 1048576});
BENCHMARK(bmGradScalerStepAndRealize)
    ->Name("optim/grad_scaler_step_and_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
