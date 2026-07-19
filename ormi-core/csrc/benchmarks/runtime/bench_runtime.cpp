#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/runtime/dispatch/device.h>

namespace {
    void bmDeviceCpu(benchmark::State& state) {
        for (auto _ : state) {
            auto device = ormi::core::Device::cpu();
            benchmark::DoNotOptimize(device);
        }
    }

    void bmDeviceFromCpu(benchmark::State& state) {
        for (auto _ : state) {
            auto device = ormi::core::Device::from("cpu");
            benchmark::DoNotOptimize(device);
        }
    }

    void bmRuntimeAllocatorCpu(benchmark::State& state) {
        const auto device = ormi::core::Device::cpu();

        for (auto _ : state) {
            const auto* allocator = ormi::core::runtime::device::allocator(device);
            benchmark::DoNotOptimize(allocator);
        }
    }

    void bmRuntimeSynchronizeCpu(benchmark::State& state) {
        const auto device = ormi::core::Device::cpu();

        for (auto _ : state) {
            ormi::core::runtime::device::synchronize(device);
        }
    }

#ifdef ORMI_ENABLE_CUDA
    void bmDeviceCuda(benchmark::State& state) {
        for (auto _ : state) {
            auto device = ormi::core::Device::cuda();
            benchmark::DoNotOptimize(device);
        }
    }

    void bmRuntimeAllocatorCuda(benchmark::State& state) {
        const auto device = ormi::core::Device::cuda();

        for (auto _ : state) {
            const auto* allocator = ormi::core::runtime::device::allocator(device);
            benchmark::DoNotOptimize(allocator);
        }
    }

    void bmRuntimeSynchronizeCuda(benchmark::State& state) {
        const auto device = ormi::core::Device::cuda();

        for (auto _ : state) {
            ormi::core::runtime::device::synchronize(device);
        }
    }
#endif
}

BENCHMARK(bmDeviceCpu)->Name("device/cpu");
BENCHMARK(bmDeviceFromCpu)->Name("device/from_cpu");
BENCHMARK(bmRuntimeAllocatorCpu)->Name("runtime/allocator_cpu");
BENCHMARK(bmRuntimeSynchronizeCpu)->Name("runtime/synchronize_cpu");

#ifdef ORMI_ENABLE_CUDA
BENCHMARK(bmDeviceCuda)->Name("device/cuda");
BENCHMARK(bmRuntimeAllocatorCuda)->Name("runtime/allocator_cuda");
BENCHMARK(bmRuntimeSynchronizeCuda)->Name("runtime/synchronize_cuda");
#endif
