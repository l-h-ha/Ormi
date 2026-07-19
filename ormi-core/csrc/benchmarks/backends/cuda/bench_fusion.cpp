#include <cstdint>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/runtime/dispatch/device.h>
#include <ormi/core/tensor.h>

namespace {
    bool require_cuda(benchmark::State& state) {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
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

    std::shared_ptr<ormi::core::Tensor> chain(
        const std::shared_ptr<ormi::core::Tensor>& input
    ) {
        return ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
    }

    void bmCUDAFusionFirstCompile(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::cuda();
        const auto input = ormi::core::ops::full({1024}, 0.5f, device);
        input->realize();

        for (auto _ : state) {
            auto output = ormi::core::ops::cos(ormi::core::ops::exp(chain(input)));
            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }
    }

    void bmCUDAFusion(benchmark::State& state, bool complete) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::cuda();
        const auto input = ormi::core::ops::full({elements}, 0.5f, device);
        input->realize();
        auto warmup = chain(input);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto output = chain(input);
            state.ResumeTiming();
            output->realize();
            if (complete) {
                ormi::core::runtime::device::synchronize(device);
            }
            benchmark::DoNotOptimize(output);
        }

        ormi::core::runtime::device::synchronize(device);
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAUnfused(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::cuda();
        const auto input = ormi::core::ops::full({elements}, 0.5f, device);
        input->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto square = ormi::core::ops::mul(input, input);
            auto sum = ormi::core::ops::add(square, input);
            auto output = ormi::core::ops::sin(sum);
            state.ResumeTiming();

            square->realize();
            sum->realize();
            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }
}

BENCHMARK(bmCUDAFusionFirstCompile)
    ->Name("runtime/cuda_fusion/first_compile_complete")
    ->Iterations(1)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusion, submit, false)
    ->Name("runtime/cuda_fusion/cached_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusion, complete, true)
    ->Name("runtime/cuda_fusion/cached_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmCUDAUnfused)
    ->Name("runtime/cuda_fusion/unfused_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();
