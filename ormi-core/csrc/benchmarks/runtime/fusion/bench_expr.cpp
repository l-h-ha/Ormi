#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/cache.h"
#include "backends/cpu/jit/codegen.h"
#include "backends/cpu/jit/kernel.h"
#include "core/runtime/fusion/expr.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> chain() {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float32
        );
        *static_cast<float*>(input->data()) = 1.0f;
        return ormi::core::ops::sin(
            ormi::core::ops::exp(
                ormi::core::ops::add(
                    ormi::core::ops::mul(input, input),
                    input
                )
            )
        );
    }

    std::shared_ptr<ormi::core::Tensor> add_chain(std::int64_t operations) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float32
        );
        auto result = input;

        for (std::int64_t operation = 0; operation < operations; ++operation) {
            result = ormi::core::ops::add(result, input);
        }
        return result;
    }

    void bmBuild(benchmark::State& state) {
        const auto root = add_chain(state.range(0));

        for (auto _ : state) {
            auto result = ormi::core::runtime::fusion::build(*root);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmEmit(benchmark::State& state) {
        const auto root = chain();
        const auto expression = ormi::core::runtime::fusion::build(*root).value();

        for (auto _ : state) {
            auto source = ormi::backends::cpu::jit::emit(expression);
            benchmark::DoNotOptimize(source);
        }
    }

    void bmCachedExecute(benchmark::State& state) {
        const auto root = chain();
        const auto expression = ormi::core::runtime::fusion::build(*root).value();
        root->realize();

        for (auto _ : state) {
            ormi::backends::cpu::jit::execute(*root, expression);
            benchmark::ClobberMemory();
        }
    }

    void unexpectedBuild(
        const std::filesystem::path&,
        const std::filesystem::path&
    ) {
        throw std::logic_error("CPU JIT disk-cache benchmark missed its artifact.");
    }

    void bmDiskLookup(benchmark::State& state) {
        const auto root = chain();
        const auto expression = ormi::core::runtime::fusion::build(*root).value();
        const auto source = ormi::backends::cpu::jit::emit(expression);
        root->realize();

        for (auto _ : state) {
            auto artifact = ormi::backends::cpu::jit::cache::get(
                source,
                unexpectedBuild
            );
            benchmark::DoNotOptimize(artifact.library);
        }
    }
}

BENCHMARK(bmBuild)
    ->Name("runtime/fusion/build")
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256);
BENCHMARK(bmEmit)->Name("runtime/fusion/emit");
BENCHMARK(bmDiskLookup)->Name("runtime/fusion/disk_lookup");
BENCHMARK(bmCachedExecute)
    ->Name("runtime/fusion/cached_execute")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);
