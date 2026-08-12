#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/memory/allocator.h>

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

    void report_size(benchmark::State& state, std::size_t bytes) {
        state.counters["bytes"] = static_cast<double>(bytes);
    }

    void bmCUDASetCurrentDevice(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        cudaError_t status = cudaSetDevice(device.index());
        if (status != cudaSuccess) {
            state.SkipWithError(cudaGetErrorString(status));
            return;
        }

        for (auto _ : state) {
            status = cudaSetDevice(device.index());
            benchmark::DoNotOptimize(status);
        }
    }

    void bmCUDAGetCurrentDevice(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        cudaError_t status = cudaSetDevice(device.index());
        if (status != cudaSuccess) {
            state.SkipWithError(cudaGetErrorString(status));
            return;
        }

        for (auto _ : state) {
            int current = -1;
            status = cudaGetDevice(&current);
            benchmark::DoNotOptimize(current);
            benchmark::DoNotOptimize(status);
        }
    }

    void bmCUDAAllocatorLookup(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        for (auto _ : state) {
            auto* allocator =
                ormi::core::runtime::device::allocator(device);
            benchmark::DoNotOptimize(allocator);
        }
    }

    void bmORMIAllocateHost(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        const auto bytes = static_cast<std::size_t>(state.range(0));
        auto* allocator = ormi::core::runtime::device::allocator(device);

        for (auto _ : state) {
            ormi::core::runtime::device::synchronize(device);
            const auto start = std::chrono::steady_clock::now();
            void* ptr = allocator->allocate(bytes);
            const auto elapsed = std::chrono::steady_clock::now() - start;
            ormi::core::runtime::device::synchronize(device);

            benchmark::DoNotOptimize(ptr);
            allocator->deallocate(ptr, bytes);
            ormi::core::runtime::device::synchronize(device);
            state.SetIterationTime(
                std::chrono::duration<double>(elapsed).count()
            );
        }

        report_size(state, bytes);
    }

    void bmORMIDeallocateHost(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        const auto bytes = static_cast<std::size_t>(state.range(0));
        auto* allocator = ormi::core::runtime::device::allocator(device);

        for (auto _ : state) {
            void* ptr = allocator->allocate(bytes);
            ormi::core::runtime::device::synchronize(device);

            const auto start = std::chrono::steady_clock::now();
            allocator->deallocate(ptr, bytes);
            const auto elapsed = std::chrono::steady_clock::now() - start;
            ormi::core::runtime::device::synchronize(device);
            state.SetIterationTime(
                std::chrono::duration<double>(elapsed).count()
            );
        }

        report_size(state, bytes);
    }

    void bmORMIPoolBurst(
        benchmark::State& benchmark_state,
        std::uint64_t release_threshold
    ) {
        if (!require_cuda(benchmark_state)) {
            return;
        }

        int pool_supported = 0;
        const cudaError_t capability_status = cudaDeviceGetAttribute(
            &pool_supported,
            cudaDevAttrMemoryPoolsSupported,
            0
        );

        if (capability_status != cudaSuccess || pool_supported == 0) {
            benchmark_state.SkipWithError("CUDA memory pools are unavailable.");
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        auto* allocator = ormi::core::runtime::device::allocator(device);
        void* probe = allocator->allocate(1);
        ormi::core::runtime::device::synchronize(device);

        CUmemoryPool allocation_pool = nullptr;
        const CUresult pool_status = cuPointerGetAttribute(
            &allocation_pool,
            CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
            reinterpret_cast<CUdeviceptr>(probe)
        );

        allocator->deallocate(probe, 1);
        ormi::core::runtime::device::synchronize(device);

        if (pool_status != CUDA_SUCCESS || !allocation_pool) {
            benchmark_state.SkipWithError("ORMI allocation did not use a CUDA memory pool.");
            return;
        }

        cudaMemPool_t pool = reinterpret_cast<cudaMemPool_t>(allocation_pool);

        std::uint64_t original_threshold = 0;
        cudaError_t status = cudaMemPoolGetAttribute(
            pool,
            cudaMemPoolAttrReleaseThreshold,
            &original_threshold
        );

        if (status != cudaSuccess) {
            benchmark_state.SkipWithError(cudaGetErrorString(status));
            return;
        }

        status = cudaMemPoolSetAttribute(
            pool,
            cudaMemPoolAttrReleaseThreshold,
            &release_threshold
        );

        if (status != cudaSuccess) {
            benchmark_state.SkipWithError(cudaGetErrorString(status));
            return;
        }

        const auto bytes = static_cast<std::size_t>(benchmark_state.range(0));
        const auto buffer_count = static_cast<std::size_t>(benchmark_state.range(1));
        std::vector<void*> pointers(buffer_count);

        allocator->empty_cache();

        for (void*& ptr : pointers) {
            ptr = allocator->allocate(bytes);
        }

        for (void* ptr : pointers) {
            allocator->deallocate(ptr, bytes);
        }

        ormi::core::runtime::device::synchronize(device);

        for (auto _ : benchmark_state) {
            for (void*& ptr : pointers) {
                ptr = allocator->allocate(bytes);
                benchmark::DoNotOptimize(ptr);
            }

            for (void* ptr : pointers) {
                allocator->deallocate(ptr, bytes);
            }

            ormi::core::runtime::device::synchronize(device);
        }

        allocator->empty_cache();
        (void)cudaMemPoolSetAttribute(
            pool,
            cudaMemPoolAttrReleaseThreshold,
            &original_threshold
        );

        benchmark_state.counters["buffer_bytes"] = static_cast<double>(bytes);
        benchmark_state.counters["buffers"] = static_cast<double>(buffer_count);
        benchmark_state.counters["burst_bytes"] = static_cast<double>(
            bytes * buffer_count
        );
    }

    void bmCUDADirectAllocateDeallocate(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto bytes = static_cast<std::size_t>(state.range(0));
        void* warmup = nullptr;
        cudaMalloc(&warmup, bytes);
        cudaFree(warmup);

        for (auto _ : state) {
            void* ptr = nullptr;
            const cudaError_t allocate_status = cudaMalloc(&ptr, bytes);

            if (allocate_status != cudaSuccess) {
                state.SkipWithError(cudaGetErrorString(allocate_status));
                break;
            }

            benchmark::DoNotOptimize(ptr);

            const cudaError_t free_status = cudaFree(ptr);

            if (free_status != cudaSuccess) {
                state.SkipWithError(cudaGetErrorString(free_status));
                break;
            }
        }

        report_size(state, bytes);
    }

    void bmORMIAllocateDeallocateSubmit(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        const auto bytes = static_cast<std::size_t>(state.range(0));
        auto* allocator = ormi::core::runtime::device::allocator(device);
        void* warmup = allocator->allocate(bytes);
        allocator->deallocate(warmup, bytes);
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            void* ptr = allocator->allocate(bytes);
            benchmark::DoNotOptimize(ptr);
            allocator->deallocate(ptr, bytes);
        }

        ormi::core::runtime::device::synchronize(device);
        report_size(state, bytes);
    }

    void bmORMIAllocateDeallocateComplete(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        const auto bytes = static_cast<std::size_t>(state.range(0));
        auto* allocator = ormi::core::runtime::device::allocator(device);
        void* warmup = allocator->allocate(bytes);
        allocator->deallocate(warmup, bytes);
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            void* ptr = allocator->allocate(bytes);
            benchmark::DoNotOptimize(ptr);
            allocator->deallocate(ptr, bytes);
            ormi::core::runtime::device::synchronize(device);
        }

        report_size(state, bytes);
    }
}

BENCHMARK(bmCUDASetCurrentDevice)
    ->Name("memory/cuda_context/set_current");

BENCHMARK(bmCUDAGetCurrentDevice)
    ->Name("memory/cuda_context/get_current");

BENCHMARK(bmCUDAAllocatorLookup)
    ->Name("memory/cuda_allocator/lookup");

BENCHMARK(bmORMIAllocateHost)
    ->Name("memory/cuda_allocator/allocate_host")
    ->Arg(64)
    ->Arg(4096)
    ->Arg(1048576)
    ->Iterations(1000)
    ->UseManualTime();

BENCHMARK(bmORMIDeallocateHost)
    ->Name("memory/cuda_allocator/deallocate_host")
    ->Arg(64)
    ->Arg(4096)
    ->Arg(1048576)
    ->Iterations(1000)
    ->UseManualTime();

BENCHMARK(bmCUDADirectAllocateDeallocate)
    ->Name("memory/cuda_alloc_free/direct")
    ->Arg(64)
    ->Arg(4096)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmORMIAllocateDeallocateSubmit)
    ->Name("memory/cuda_alloc_free/ormi_submit")
    ->Arg(64)
    ->Arg(4096)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmORMIAllocateDeallocateComplete)
    ->Name("memory/cuda_alloc_free/ormi_complete")
    ->Arg(64)
    ->Arg(4096)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmORMIPoolBurst, release, std::uint64_t{0})
    ->Name("memory/cuda_pool_burst/release")
    ->Args({4096, 1})
    ->Args({4096, 16})
    ->Args({4096, 256})
    ->Args({1048576, 1})
    ->Args({1048576, 16})
    ->Args({1048576, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmORMIPoolBurst, retain_64mib, std::uint64_t{64} * 1024 * 1024)
    ->Name("memory/cuda_pool_burst/retain_64mib")
    ->Args({4096, 1})
    ->Args({4096, 16})
    ->Args({4096, 256})
    ->Args({1048576, 1})
    ->Args({1048576, 16})
    ->Args({1048576, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmORMIPoolBurst,
    retain_unlimited,
    std::numeric_limits<std::uint64_t>::max()
)
    ->Name("memory/cuda_pool_burst/retain_unlimited")
    ->Args({4096, 1})
    ->Args({4096, 16})
    ->Args({4096, 256})
    ->Args({1048576, 1})
    ->Args({1048576, 16})
    ->Args({1048576, 256})
    ->UseRealTime();
