#include <cstddef>
#include <cstdint>
#include <limits>
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/optim/grad_scaler.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"

namespace {
    bool requireCUDA(benchmark::State& state) {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);

        if (status != cudaSuccess) {
            state.SkipWithError(cudaGetErrorString(status));
            return false;
        }

        if (device_count == 0) {
            state.SkipWithError("No CUDA device is available.");
            return false;
        }

        return true;
    }

    void bmCUDASGD(benchmark::State& state, bool scaled, bool complete) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto parameter = ormi::core::ops::full({elements}, 1.0f, device);
        const auto gradient = ormi::core::ops::full(
            {elements},
            scaled ? 256.0f : 0.25f,
            device
        );
        parameter->realize();
        gradient->realize();

        auto warmup = ormi::core::optim::sgd(
            parameter,
            gradient,
            0.001,
            scaled ? 1024.0 : 1.0
        );
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto next = ormi::core::optim::sgd(
                parameter,
                gradient,
                0.001,
                scaled ? 1024.0 : 1.0
            );
            state.ResumeTiming();

            next->realize();
            benchmark::DoNotOptimize(next);

            if (complete) {
                ormi::core::runtime::device::synchronize(device);
            }
        }

        ormi::core::runtime::device::synchronize(device);

        if (complete) {
            state.SetItemsProcessed(state.iterations() * elements);
        }
    }

    void bmCUDADecomposedScaledSGD(benchmark::State& state) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto parameter = ormi::core::ops::full({elements}, 1.0f, device);
        const auto gradient = ormi::core::ops::full({elements}, 256.0f, device);
        parameter->realize();
        gradient->realize();

        for (auto _ : state) {
            state.PauseTiming();
            const auto factor = ormi::core::ops::full(
                ormi::core::Shape{},
                0.001f / 1024.0f,
                device
            );
            auto next = ormi::core::ops::sub(
                parameter,
                ormi::core::ops::mul(gradient, factor)
            );
            state.ResumeTiming();

            next->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(next);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAGradScalerStep(benchmark::State& state) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto parameter = ormi::core::ops::full({elements}, 1.0f, device);
        const auto gradient = ormi::core::ops::full({elements}, 256.0f, device);
        ormi::core::optim::GradScaler scaler(
            1024.0,
            ormi::core::optim::ScalePolicy{
                .interval = std::numeric_limits<std::size_t>::max()
            }
        );
        parameter->realize();
        gradient->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            auto next = scaler.step(parameter, gradient, 0.001);
            (*next)->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(next);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }
}

BENCHMARK_CAPTURE(bmCUDASGD, submit, false, false)
    ->Name("optim/cuda_sgd/submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDASGD, complete, false, true)
    ->Name("optim/cuda_sgd/complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDASGD, scaled_complete, true, true)
    ->Name("optim/cuda_sgd/scaled_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmCUDADecomposedScaledSGD)
    ->Name("optim/cuda_sgd/decomposed_scaled_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmCUDAGradScalerStep)
    ->Name("optim/cuda_sgd/grad_scaler_step_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();
