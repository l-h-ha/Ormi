#include <cstdint>
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"

namespace {
    bool require_pointwise_cuda(benchmark::State& state) {
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

    void bmCUDAAdd(benchmark::State& state, bool broadcast, bool complete) {
        if (!require_pointwise_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto lhs = ormi::core::ops::full({elements}, 1.25f, device);
        const auto rhs = ormi::core::ops::full(
            broadcast ? ormi::core::Shape{} : ormi::core::Shape{elements},
            2.0f,
            device
        );
        lhs->realize();
        rhs->realize();

        auto warmup = ormi::core::ops::add(lhs, rhs);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::add(lhs, rhs);
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

    void bmCUDANeg(benchmark::State& state, bool complete) {
        if (!require_pointwise_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto input = ormi::core::ops::full({elements}, 1.25f, device);
        input->realize();

        auto warmup = ormi::core::ops::neg(input);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::neg(input);
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

    void bmCUDABroadcastGrid(benchmark::State& state, bool complete) {
        if (!require_pointwise_cuda(state)) {
            return;
        }

        const auto side = state.range(0);
        const auto elements = side * side;
        const auto device = ormi::core::Device::from("cuda");
        const auto lhs = ormi::core::ops::full({side, 1}, 1.25f, device);
        const auto rhs = ormi::core::ops::full({1, side}, 2.0f, device);
        lhs->realize();
        rhs->realize();

        auto warmup = ormi::core::ops::add(lhs, rhs);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::add(lhs, rhs);
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

BENCHMARK_CAPTURE(bmCUDAAdd, contiguous_submit, false, false)
    ->Name("graph/cuda_pointwise_add/contiguous_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAdd, contiguous_complete, false, true)
    ->Name("graph/cuda_pointwise_add/contiguous_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAdd, broadcast_submit, true, false)
    ->Name("graph/cuda_pointwise_add/broadcast_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAdd, broadcast_complete, true, true)
    ->Name("graph/cuda_pointwise_add/broadcast_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDANeg, submit, false)
    ->Name("graph/cuda_pointwise_neg/submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDANeg, complete, true)
    ->Name("graph/cuda_pointwise_neg/complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDABroadcastGrid, submit, false)
    ->Name("graph/cuda_pointwise_add/broadcast_grid_submit")
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDABroadcastGrid, complete, true)
    ->Name("graph/cuda_pointwise_add/broadcast_grid_complete")
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();
