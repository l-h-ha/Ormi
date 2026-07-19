#include <cstddef>
#include <cstdint>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/runtime/dispatch/device.h>
#include <ormi/core/tensor.h>

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

    void report_elements(benchmark::State& state, std::size_t elements) {
        state.SetItemsProcessed(
            static_cast<std::int64_t>(state.iterations()) *
            static_cast<std::int64_t>(elements)
        );
        state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations()) *
            static_cast<std::int64_t>(elements * sizeof(float))
        );
    }

    void bmCUDAFillScalarFloat32(benchmark::State& state, bool complete) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = static_cast<std::size_t>(state.range(0));
        const auto device = ormi::core::Device::cuda();
        ormi::core::memory::Buffer buffer(device, elements * sizeof(float));
        constexpr float value = 2.5f;

        ormi::core::memory::fill_scalar(
            buffer.data(),
            device,
            &value,
            elements,
            ormi::core::DType::float32
        );
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            ormi::core::memory::fill_scalar(
                buffer.data(),
                device,
                &value,
                elements,
                ormi::core::DType::float32
            );

            if (complete) {
                ormi::core::runtime::device::synchronize(device);
            }
        }

        ormi::core::runtime::device::synchronize(device);
        report_elements(state, elements);
    }

    void bmCUDAFullFloat32(benchmark::State& state, float value, bool complete) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = static_cast<std::size_t>(state.range(0));

        for (auto _ : state) {
            state.PauseTiming();
            auto tensor = ormi::core::ops::full(
                {static_cast<std::int64_t>(elements)},
                value,
                ormi::core::Device::cuda()
            );
            state.ResumeTiming();

            tensor->realize();
            benchmark::DoNotOptimize(tensor);

            if (complete) {
                ormi::core::runtime::device::synchronize(tensor->device());
            }
        }

        ormi::core::runtime::device::synchronize(ormi::core::Device::cuda());
        report_elements(state, elements);
    }
}

BENCHMARK_CAPTURE(bmCUDAFillScalarFloat32, submit, false)
    ->Name("memory/cuda_fill_scalar_float32/submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFillScalarFloat32, complete, true)
    ->Name("memory/cuda_fill_scalar_float32/complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFullFloat32, nonzero_submit, 2.5f, false)
    ->Name("graph/cuda_full_float32/nonzero_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFullFloat32, nonzero_complete, 2.5f, true)
    ->Name("graph/cuda_full_float32/nonzero_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFullFloat32, zero_submit, 0.0f, false)
    ->Name("graph/cuda_full_float32/zero_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFullFloat32, zero_complete, 0.0f, true)
    ->Name("graph/cuda_full_float32/zero_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();
