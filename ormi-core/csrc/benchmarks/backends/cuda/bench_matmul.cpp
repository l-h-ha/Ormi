#include <cstdint>
#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"

namespace {
    bool require_cuda(benchmark::State& state) {
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

    void bmCUDAMatmul(benchmark::State& state, bool complete) {
        if (!require_cuda(state)) {
            return;
        }

        const auto batches = state.range(0);
        const auto side = state.range(1);
        const auto device = ormi::core::Device::from("cuda");
        const auto shape = ormi::core::Shape{batches, side, side};
        const auto lhs = ormi::core::ops::full(shape, 1.0f, device);
        const auto rhs = ormi::core::ops::full(shape, 1.0f, device);
        lhs->realize();
        rhs->realize();

        auto warmup = ormi::core::ops::matmul(lhs, rhs);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::matmul(lhs, rhs);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);

            if (complete) {
                ormi::core::runtime::device::synchronize(device);
            }
        }

        ormi::core::runtime::device::synchronize(device);

        if (complete) {
            const auto operations =
                2.0 * static_cast<double>(batches) *
                static_cast<double>(side) *
                static_cast<double>(side) *
                static_cast<double>(side);
            state.counters["FLOP/s"] = benchmark::Counter(
                operations * static_cast<double>(state.iterations()),
                benchmark::Counter::kIsRate
            );
        }
    }

    void bmCUDAIrregularMatmul(benchmark::State& state, bool complete) {
        if (!require_cuda(state)) {
            return;
        }

        const auto batch_axis = state.range(0);
        const auto side = state.range(1);
        const auto device = ormi::core::Device::from("cuda");
        const auto lhs = ormi::core::ops::full(
            {batch_axis, 1, side, side},
            1.0f,
            device
        );
        const auto rhs = ormi::core::ops::full(
            {1, batch_axis, side, side},
            1.0f,
            device
        );
        lhs->realize();
        rhs->realize();

        auto warmup = ormi::core::ops::matmul(lhs, rhs);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::matmul(lhs, rhs);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);

            if (complete) {
                ormi::core::runtime::device::synchronize(device);
            }
        }

        ormi::core::runtime::device::synchronize(device);
    }
}

BENCHMARK_CAPTURE(bmCUDAMatmul, submit, false)
    ->Name("graph/cuda_matmul_float32/submit")
    ->Args({1, 16})
    ->Args({1, 64})
    ->Args({1, 256})
    ->Args({256, 16})
    ->Args({1024, 4})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAMatmul, complete, true)
    ->Name("graph/cuda_matmul_float32/complete")
    ->Args({1, 16})
    ->Args({1, 64})
    ->Args({1, 256})
    ->Args({256, 16})
    ->Args({1024, 4})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAIrregularMatmul, submit, false)
    ->Name("graph/cuda_matmul_float32/irregular_submit")
    ->Args({16, 16})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAIrregularMatmul, complete, true)
    ->Name("graph/cuda_matmul_float32/irregular_complete")
    ->Args({16, 16})
    ->UseRealTime();
