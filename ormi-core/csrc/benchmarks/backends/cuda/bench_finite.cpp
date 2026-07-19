#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/runtime/dispatch/device.h>
#include <ormi/core/runtime/dispatch/finite.h>
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

    void bmCUDAFiniteQuery(benchmark::State& state) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::cuda();
        const auto input = ormi::core::ops::full({elements}, 1.0f, device);
        const std::array<const ormi::core::Tensor*, 1> inputs{input.get()};
        input->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            const bool finite = ormi::core::runtime::finite::all(inputs);
            benchmark::DoNotOptimize(finite);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDALazyFinite(benchmark::State& state) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::cuda();
        const auto input = ormi::core::ops::full({elements}, 1.0f, device);
        input->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto result = ormi::core::ops::all_finite(input);
            state.ResumeTiming();

            result->realize();
            benchmark::DoNotOptimize(result);
        }

        ormi::core::runtime::device::synchronize(device);
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAFiniteBatch(benchmark::State& state) {
        if (!requireCUDA(state)) {
            return;
        }

        const auto count = static_cast<std::size_t>(state.range(0));
        const auto elements = state.range(1);
        const auto device = ormi::core::Device::cuda();
        std::vector<std::shared_ptr<ormi::core::Tensor>> tensors;
        std::vector<const ormi::core::Tensor*> inputs;
        tensors.reserve(count);
        inputs.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            tensors.push_back(ormi::core::ops::full({elements}, 1.0f, device));
            tensors.back()->realize();
            inputs.push_back(tensors.back().get());
        }

        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            const bool finite = ormi::core::runtime::finite::all(inputs);
            benchmark::DoNotOptimize(finite);
        }

        state.SetItemsProcessed(state.iterations() * count * elements);
    }
}

BENCHMARK(bmCUDAFiniteQuery)
    ->Name("graph/cuda_all_finite/direct_query")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmCUDALazyFinite)
    ->Name("graph/cuda_all_finite/lazy_realize")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmCUDAFiniteBatch)
    ->Name("graph/cuda_all_finite/batch_query")
    ->Args({1, 1024})
    ->Args({16, 1024})
    ->Args({1, 1048576})
    ->UseRealTime();
