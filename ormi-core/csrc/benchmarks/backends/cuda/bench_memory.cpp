#include <cstddef>
#include <cstdint>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>

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

    void report_bytes(benchmark::State& state, std::size_t bytes) {
        state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations()) *
            static_cast<std::int64_t>(bytes)
        );
    }

    void bmCUDAHostToDevice(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto bytes = static_cast<std::size_t>(state.range(0));
        ormi::core::memory::Buffer host(ormi::core::Device::cpu(), bytes);
        ormi::core::memory::Buffer device(ormi::core::Device::cuda(), bytes);
        ormi::core::memory::copy(device, host);

        for (auto _ : state) {
            ormi::core::memory::copy(device, host);
        }

        report_bytes(state, bytes);
    }

    void bmCUDADeviceToHost(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto bytes = static_cast<std::size_t>(state.range(0));
        ormi::core::memory::Buffer host(ormi::core::Device::cpu(), bytes);
        ormi::core::memory::Buffer device(ormi::core::Device::cuda(), bytes);
        ormi::core::memory::copy(device, host);

        for (auto _ : state) {
            ormi::core::memory::copy(host, device);
        }

        report_bytes(state, bytes);
    }

    void bmCUDADeviceToDevice(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto bytes = static_cast<std::size_t>(state.range(0));
        ormi::core::memory::Buffer source(ormi::core::Device::cuda(), bytes);
        ormi::core::memory::Buffer destination(ormi::core::Device::cuda(), bytes);
        ormi::core::memory::fill(source, 0x5a);
        ormi::core::memory::copy(destination, source);

        for (auto _ : state) {
            ormi::core::memory::copy(destination, source);
        }

        report_bytes(state, bytes);
    }
}

BENCHMARK(bmCUDAHostToDevice)
    ->Name("memory/cuda_copy/host_to_device")
    ->UseRealTime()
    ->Arg(4096)
    ->Arg(1048576)
    ->Arg(16777216);

BENCHMARK(bmCUDADeviceToHost)
    ->Name("memory/cuda_copy/device_to_host")
    ->UseRealTime()
    ->Arg(4096)
    ->Arg(1048576)
    ->Arg(16777216);

BENCHMARK(bmCUDADeviceToDevice)
    ->Name("memory/cuda_copy/device_to_device")
    ->UseRealTime()
    ->Arg(4096)
    ->Arg(1048576)
    ->Arg(16777216);
