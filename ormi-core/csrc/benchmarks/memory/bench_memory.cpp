#include <cstddef>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/runtime/dispatch/device.h>

namespace {
    void bmCpuAllocateDeallocate64(benchmark::State& state) {
        auto* allocator = ormi::core::runtime::device::allocator(ormi::core::Device::cpu());

        for (auto _ : state) {
            void* ptr = allocator->allocate(64);
            benchmark::DoNotOptimize(ptr);
            allocator->deallocate(ptr, 64);
        }
    }

    void bmCpuAllocateDeallocate4096(benchmark::State& state) {
        auto* allocator = ormi::core::runtime::device::allocator(ormi::core::Device::cpu());

        for (auto _ : state) {
            void* ptr = allocator->allocate(4096);
            benchmark::DoNotOptimize(ptr);
            allocator->deallocate(ptr, 4096);
        }
    }

    void bmCpuBuffer64(benchmark::State& state) {
        const auto device = ormi::core::Device::cpu();

        for (auto _ : state) {
            ormi::core::memory::Buffer buffer(device, 64);
            benchmark::DoNotOptimize(buffer.data());
        }
    }

    void bmCpuZeroBytes(benchmark::State& state) {
        const auto device = ormi::core::Device::cpu();
        const auto elements = static_cast<std::size_t>(state.range(0));
        const auto bytes = elements * sizeof(float);
        ormi::core::memory::Buffer buffer(device, bytes);

        for (auto _ : state) {
            ormi::core::memory::zero(buffer);
            benchmark::ClobberMemory();
        }

        state.SetBytesProcessed(state.iterations() * bytes);
    }

    void bmCpuFillScalarFloat32Zero(benchmark::State& state) {
        const auto device = ormi::core::Device::cpu();
        const auto elements = static_cast<std::size_t>(state.range(0));
        const auto bytes = elements * sizeof(float);
        constexpr float zero = 0.0f;
        ormi::core::memory::Buffer buffer(device, bytes);

        for (auto _ : state) {
            ormi::core::memory::fill_scalar(
                buffer.data(),
                device,
                &zero,
                elements,
                ormi::core::DType::float32
            );
            benchmark::ClobberMemory();
        }

        state.SetBytesProcessed(state.iterations() * bytes);
    }
}

BENCHMARK(bmCpuAllocateDeallocate64)->Name("memory/cpu_alloc_free/64");
BENCHMARK(bmCpuAllocateDeallocate4096)->Name("memory/cpu_alloc_free/4096");
BENCHMARK(bmCpuBuffer64)->Name("memory/cpu_buffer/64");
BENCHMARK(bmCpuZeroBytes)
    ->Name("memory/cpu_zero_bytes")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
BENCHMARK(bmCpuFillScalarFloat32Zero)
    ->Name("memory/cpu_fill_scalar_float32_zero")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);
