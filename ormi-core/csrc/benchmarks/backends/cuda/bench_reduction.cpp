#include <cstdint>
#include <vector>
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/reduction.h>
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

    void bmCUDAGlobalSum(benchmark::State& state, bool complete) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto input = ormi::core::ops::full({elements}, 1.0f, device);
        input->realize();

        auto warmup = ormi::core::ops::sum(input);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::sum(input);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);

            if (complete) {
                ormi::core::runtime::device::synchronize(device);
            }
        }

        ormi::core::runtime::device::synchronize(device);

        if (complete) {
            state.SetItemsProcessed(state.iterations() * elements);
        }
    }

    void bmCUDAAxisSum(benchmark::State& state, bool first_axis, bool complete) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto side = state.range(0);
        const auto elements = side * side;
        const auto device = ormi::core::Device::from("cuda");
        const auto input = ormi::core::ops::full({side, side}, 1.0f, device);
        const std::vector<std::int64_t> axes{first_axis ? 0 : 1};
        input->realize();

        auto warmup = ormi::core::ops::sum(input, axes);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::sum(input, axes);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);

            if (complete) {
                ormi::core::runtime::device::synchronize(device);
            }
        }

        ormi::core::runtime::device::synchronize(device);

        if (complete) {
            state.SetItemsProcessed(state.iterations() * elements);
        }
    }
}

BENCHMARK_CAPTURE(bmCUDAGlobalSum, submit, false)
    ->Name("graph/cuda_sum/global_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAGlobalSum, complete, true)
    ->Name("graph/cuda_sum/global_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAxisSum, last_axis_submit, false, false)
    ->Name("graph/cuda_sum/last_axis_submit")
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAxisSum, last_axis_complete, false, true)
    ->Name("graph/cuda_sum/last_axis_complete")
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAxisSum, first_axis_submit, true, false)
    ->Name("graph/cuda_sum/first_axis_submit")
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAxisSum, first_axis_complete, true, true)
    ->Name("graph/cuda_sum/first_axis_complete")
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();
