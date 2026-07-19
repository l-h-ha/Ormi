#include <cstdint>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/runtime/dispatch/device.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

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

    void bmCUDAFloatCast(benchmark::State& state, bool reverse, bool complete) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::cuda();
        const auto input = ormi::core::ops::full({elements}, 1.25f, device);
        input->realize();

        std::shared_ptr<ormi::core::Tensor> source = input;
        auto output_dtype = ormi::core::DType::float16;

        if (reverse) {
            source = ormi::core::ops::cast(input, ormi::core::DType::float16);
            source->realize();
            output_dtype = ormi::core::DType::float32;
        }

        auto warmup = ormi::core::ops::cast(source, output_dtype);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::cast(source, output_dtype);
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

    void bmCUDACheckedIntCast(benchmark::State& state) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::cuda();
        const auto input = ormi::core::ops::full({elements}, 1.25f, device);
        input->realize();

        auto warmup = ormi::core::ops::cast(input, ormi::core::DType::int32);
        warmup->realize();

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::cast(input, ormi::core::DType::int32);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }
}

BENCHMARK_CAPTURE(bmCUDAFloatCast, float32_to_float16_submit, false, false)
    ->Name("graph/cuda_cast/float32_to_float16_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFloatCast, float32_to_float16_complete, false, true)
    ->Name("graph/cuda_cast/float32_to_float16_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFloatCast, float16_to_float32_submit, true, false)
    ->Name("graph/cuda_cast/float16_to_float32_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFloatCast, float16_to_float32_complete, true, true)
    ->Name("graph/cuda_cast/float16_to_float32_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmCUDACheckedIntCast)
    ->Name("graph/cuda_cast/float32_to_int32_checked")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();
