#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/cache.h"
#include "backends/cpu/jit/codegen.h"
#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/execute.h"
#include "backends/cpu/jit/kernel.h"
#include "backends/cpu/jit/math.h"
#include "backends/cpu/jit/target.h"
#include "core/runtime/fusion/expr.h"
#include "platform/platform.h"

namespace {
    using Clock = std::chrono::steady_clock;

    std::atomic<std::uint64_t> artifact_sequence{
        static_cast<std::uint64_t>(Clock::now().time_since_epoch().count())
    };

    struct BuildTiming {
        const ormi::backends::cpu::jit::Toolchain* compiler;
        ormi::backends::cpu::jit::Target target;
        ormi::backends::cpu::jit::Math math;
        Clock::duration elapsed{};
    };

    thread_local BuildTiming* active_build = nullptr;

    enum class ColdKernel {
        arithmetic,
        scalar_math,
        vector_math,
        reduction,
        float16
    };

    class BuildScope final {
    public:
        explicit BuildScope(BuildTiming& timing) {
            active_build = &timing;
        }

        ~BuildScope() {
            active_build = nullptr;
        }

        BuildScope(const BuildScope&) = delete;
        BuildScope& operator=(const BuildScope&) = delete;
    };

    std::shared_ptr<ormi::core::Tensor> chain(
        std::size_t extra_additions = 0,
        std::size_t elements = 1
    ) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{static_cast<std::int64_t>(elements)},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < elements; ++i) {
            values[i] = 1.0f;
        }

        auto result = ormi::core::ops::exp(
            ormi::core::ops::add(
                ormi::core::ops::mul(input, input),
                input
            )
        );
        for (std::size_t i = 0; i < extra_additions; ++i) {
            result = ormi::core::ops::add(result, input);
        }
        return ormi::core::ops::sin(result);
    }

    std::shared_ptr<ormi::core::Tensor> arithmetic(
        ormi::core::DType dtype,
        std::size_t elements,
        std::size_t extra_additions
    ) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{static_cast<std::int64_t>(elements)},
            dtype
        );
        auto result = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        for (std::size_t i = 0; i < extra_additions; ++i) {
            result = ormi::core::ops::add(result, input);
        }
        return result;
    }

    std::shared_ptr<ormi::core::Tensor> coldRoot(ColdKernel kernel) {
        constexpr std::size_t vector_elements = 1024;

        switch (kernel) {
            case ColdKernel::arithmetic:
                return arithmetic(
                    ormi::core::DType::float32,
                    vector_elements,
                    9
                );
            case ColdKernel::scalar_math:
                return chain(7);
            case ColdKernel::vector_math:
                return chain(7, vector_elements);
            case ColdKernel::reduction:
                return ormi::core::ops::sum(
                    arithmetic(
                        ormi::core::DType::float32,
                        vector_elements,
                        9
                    )
                );
            case ColdKernel::float16:
                return arithmetic(
                    ormi::core::DType::float16,
                    vector_elements,
                    9
                );
        }

        throw std::logic_error("Unknown cold CPU JIT benchmark kernel.");
    }

    void writeBenchmarkArtifact(
        const std::filesystem::path&,
        const std::filesystem::path& output
    ) {
        constexpr std::string_view contents = "ORMI CPU JIT cache benchmark";
        std::ofstream file(output, std::ios::binary | std::ios::trunc);
        file.write(
            contents.data(),
            static_cast<std::streamsize>(contents.size())
        );
        if (!file) {
            throw std::runtime_error("Cannot write CPU JIT benchmark artifact.");
        }
    }

    void compileArtifact(
        const std::filesystem::path& source,
        const std::filesystem::path& output
    ) {
        const auto& compiler = ormi::backends::cpu::jit::toolchain();
        ormi::backends::cpu::jit::build(
            compiler,
            ormi::backends::cpu::jit::selected_target(),
            ormi::backends::cpu::jit::selected_math(compiler.id),
            source,
            output
        );
    }

    void profileBuild(
        const std::filesystem::path& source,
        const std::filesystem::path& output
    ) {
        if (!active_build) {
            throw std::logic_error("CPU JIT build timing is not active.");
        }

        const auto started = Clock::now();
        ormi::backends::cpu::jit::build(
            *active_build->compiler,
            active_build->target,
            active_build->math,
            source,
            output
        );
        active_build->elapsed = Clock::now() - started;
    }

    double microseconds(Clock::duration duration) {
        return std::chrono::duration<double, std::micro>(duration).count();
    }

    void discardArtifacts(std::string_view source) {
        using namespace ormi::backends::cpu::jit;

        cache::discard(source, context(math(MathABI::direct)));
        cache::discard(source, context(math(MathABI::buffered)));
        cache::discard(source, context(Math{}));
    }

    void bmCompilerResolve(benchmark::State& state) {
        for (auto _ : state) {
            const auto compiler = ormi::backends::cpu::jit::resolve();
            benchmark::DoNotOptimize(compiler.executable.native().data());
        }
    }

    void bmCompilerStartup(benchmark::State& state) {
        const auto& compiler = ormi::backends::cpu::jit::toolchain();
        const std::span<const std::string> arguments{};

        for (auto _ : state) {
            auto result = ormi::backends::cpu::jit::run(
                compiler,
                arguments
            );
            benchmark::DoNotOptimize(result.code);
            benchmark::DoNotOptimize(result.output.data());
        }

        state.SetLabel(std::string(
            ormi::backends::cpu::jit::to_string(compiler.id)
        ));
    }

    void bmColdKernel(benchmark::State& state, ColdKernel kernel) {
        using namespace ormi::backends::cpu::jit;

        const auto root = coldRoot(kernel);
        const auto expression = ormi::core::runtime::fusion::build(*root).value();
        const auto& compiler = toolchain();
        const auto target = selected_target();
        const auto functions = selected_math(compiler.id);
        const auto identity = context(functions);

        for (auto _ : state) {
            const auto started = Clock::now();

            const auto loop_started = Clock::now();
            const auto loop = plan(expression);
            const auto loop_finished = Clock::now();

            const auto emit_started = Clock::now();
            auto source = emit(expression, loop);
            source += "\n// benchmark artifact ";
            source += std::to_string(
                artifact_sequence.fetch_add(1, std::memory_order_relaxed)
            );
            const auto emit_finished = Clock::now();

            const auto context_started = Clock::now();
            BuildTiming build_timing{
                .compiler = &compiler,
                .target = target,
                .math = functions
            };
            const auto context_finished = Clock::now();
            const auto cache_started = context_finished;
            Clock::time_point cache_finished;
            Clock::time_point load_started;
            Clock::time_point load_finished;
            bool cache_hit = false;
            {
                auto artifact = [&] {
                    BuildScope scope(build_timing);
                    return cache::get(source, identity, profileBuild);
                }();
                cache_finished = Clock::now();
                load_started = cache_finished;
                ormi::platform::Library library(artifact.library);
                auto* function = library.symbol("ormi_kernel");
                benchmark::DoNotOptimize(function);
                load_finished = Clock::now();
                cache_hit = artifact.cached;
            }

            const auto cache_elapsed =
                cache_finished - cache_started - build_timing.elapsed;
            state.counters["loop_us"] = microseconds(
                loop_finished - loop_started
            );
            state.counters["emit_us"] = microseconds(
                emit_finished - emit_started
            );
            state.counters["context_us"] = microseconds(
                context_finished - context_started
            );
            state.counters["compile_link_us"] = microseconds(
                build_timing.elapsed
            );
            state.counters["cache_us"] = microseconds(cache_elapsed);
            state.counters["load_us"] = microseconds(
                load_finished - load_started
            );
            state.counters["source_bytes"] =
                static_cast<double>(source.size());
            state.counters["operations"] =
                static_cast<double>(expression.operations);
            state.counters["cmath"] =
                source.find("#include <cmath>") != std::string::npos ? 1.0 : 0.0;
            state.counters["immintrin"] =
                source.find("#include <immintrin.h>") != std::string::npos
                    ? 1.0
                    : 0.0;
            state.counters["float16_headers"] =
                source.find("#include <bit>") != std::string::npos ? 1.0 : 0.0;
            state.counters["cache_hit"] = cache_hit ? 1.0 : 0.0;
            state.SetLabel(std::string(to_string(compiler.id)));
            state.SetIterationTime(
                std::chrono::duration<double>(load_finished - started).count()
            );
            discardArtifacts(source);
        }
    }

    void bmMemoryLookup(benchmark::State& state) {
        const auto root = chain();
        const auto expression = ormi::core::runtime::fusion::build(*root).value();
        const auto compiled = ormi::backends::cpu::jit::compile(expression);
        benchmark::DoNotOptimize(compiled.get());

        for (auto _ : state) {
            const auto current = ormi::backends::cpu::jit::compile(expression);
            benchmark::DoNotOptimize(current.get());
        }
    }

    void bmAlternatingMemoryLookup(benchmark::State& state) {
        const auto first_root = chain();
        const auto second_root = chain(2);
        const auto first = ormi::core::runtime::fusion::build(
            *first_root
        ).value();
        const auto second = ormi::core::runtime::fusion::build(
            *second_root
        ).value();
        benchmark::DoNotOptimize(ormi::backends::cpu::jit::compile(first).get());
        benchmark::DoNotOptimize(ormi::backends::cpu::jit::compile(second).get());
        std::size_t iteration = 0;

        for (auto _ : state) {
            const auto& expression = iteration++ % 2 == 0 ? first : second;
            const auto compiled = ormi::backends::cpu::jit::compile(expression);
            benchmark::DoNotOptimize(compiled.get());
        }
    }

    void bmBatchMemoryLookup(benchmark::State& state) {
        const auto count = static_cast<std::size_t>(state.range(0));
        std::vector<std::shared_ptr<ormi::core::Tensor>> roots;
        std::vector<ormi::core::runtime::fusion::Expr> expressions;
        std::vector<const ormi::core::runtime::fusion::Expr*> pointers;
        roots.reserve(count);
        expressions.reserve(count);
        pointers.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            roots.push_back(chain(i + 3));
            expressions.push_back(
                ormi::core::runtime::fusion::build(*roots.back()).value()
            );
        }
        for (const auto& expression : expressions) {
            pointers.push_back(&expression);
        }
        benchmark::DoNotOptimize(
            ormi::backends::cpu::jit::compile(pointers).data()
        );

        for (auto _ : state) {
            const auto compiled = ormi::backends::cpu::jit::compile(pointers);
            benchmark::DoNotOptimize(compiled.data());
        }
        state.SetItemsProcessed(
            state.iterations() * static_cast<std::int64_t>(count)
        );
    }

    void bmDiskLookup(benchmark::State& state) {
        constexpr std::string_view source =
            "ORMI CPU JIT cache lookup benchmark source";
        constexpr std::string_view context = "benchmark";
        ormi::backends::cpu::jit::cache::discard(source, context);

        {
            const auto seeded = ormi::backends::cpu::jit::cache::get(
                source,
                context,
                writeBenchmarkArtifact
            );
            auto seeded_from_cache = seeded.cached;
            benchmark::DoNotOptimize(seeded_from_cache);

            for (auto _ : state) {
                const auto artifact = ormi::backends::cpu::jit::cache::get(
                    source,
                    context,
                    writeBenchmarkArtifact
                );
                auto from_cache = artifact.cached;
                benchmark::DoNotOptimize(from_cache);
            }
        }

        ormi::backends::cpu::jit::cache::discard(source, context);
    }

    void bmDiskFind(benchmark::State& state, bool hit) {
        const auto source = "ORMI CPU JIT find benchmark " +
            std::to_string(artifact_sequence.fetch_add(1));
        constexpr std::string_view context = "benchmark";
        const auto seeded = ormi::backends::cpu::jit::cache::get(
            source, context, writeBenchmarkArtifact
        );
        const auto requested = hit ? source : source + " missing";

        for (auto _ : state) {
            const auto found = ormi::backends::cpu::jit::cache::find(requested, context);
            benchmark::DoNotOptimize(found.has_value());
            if (found.has_value() != hit) {
                state.SkipWithError("CPU JIT disk lookup returned an unexpected result.");
                break;
            }
        }
    }

    void bmLibraryLoad(benchmark::State& state) {
        const auto root = chain(1);
        const auto expression = ormi::core::runtime::fusion::build(*root).value();
        const auto source = ormi::backends::cpu::jit::emit(expression);
        const auto& compiler = ormi::backends::cpu::jit::toolchain();
        const auto identity = ormi::backends::cpu::jit::context(
            ormi::backends::cpu::jit::selected_math(compiler.id)
        );
        const auto artifact = ormi::backends::cpu::jit::cache::get(
            source,
            identity,
            compileArtifact
        );

        for (auto _ : state) {
            ormi::platform::Library library(artifact.library);
            auto* function = library.symbol("ormi_kernel");
            benchmark::DoNotOptimize(function);
        }
    }

    void bmPrepare(benchmark::State& state) {
        const auto root = chain();
        const auto expression = ormi::core::runtime::fusion::build(*root).value();
        const auto compiled = ormi::backends::cpu::jit::compile(expression);
        const std::span<ormi::core::Tensor* const> auxiliary_outputs;

        for (auto _ : state) {
            auto setup = ormi::backends::cpu::jit::execution::prepare(
                auxiliary_outputs,
                expression,
                compiled
            );
            benchmark::DoNotOptimize(setup.buffers());
        }
    }

    void bmPreparedExecute(benchmark::State& state) {
        const auto root = chain();
        const auto expression = ormi::core::runtime::fusion::build(*root).value();
        root->realize();
        auto setup = ormi::backends::cpu::jit::prepare(*root, expression);

        for (auto _ : state) {
            setup.run(*root);
            benchmark::ClobberMemory();
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

    void bmEmit(benchmark::State& state) {
        const auto root = chain();
        const auto expression = ormi::core::runtime::fusion::build(*root).value();

        for (auto _ : state) {
            auto source = ormi::backends::cpu::jit::emit(expression);
            benchmark::DoNotOptimize(source);
        }
    }
}

BENCHMARK(bmCompilerResolve)
    ->Name("runtime/jit/compiler_resolve")
    ->UseRealTime();
BENCHMARK(bmCompilerStartup)
    ->Name("runtime/jit/compiler_startup")
    ->UseRealTime();
BENCHMARK_CAPTURE(bmColdKernel, arithmetic, ColdKernel::arithmetic)
    ->Name("runtime/jit/cold_kernel/arithmetic")
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK_CAPTURE(bmColdKernel, scalar_math, ColdKernel::scalar_math)
    ->Name("runtime/jit/cold_kernel/scalar_math")
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK_CAPTURE(bmColdKernel, vector_math, ColdKernel::vector_math)
    ->Name("runtime/jit/cold_kernel/vector_math")
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK_CAPTURE(bmColdKernel, reduction, ColdKernel::reduction)
    ->Name("runtime/jit/cold_kernel/reduction")
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK_CAPTURE(bmColdKernel, float16, ColdKernel::float16)
    ->Name("runtime/jit/cold_kernel/float16")
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(bmMemoryLookup)
    ->Name("runtime/jit/memory_lookup")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);
BENCHMARK(bmAlternatingMemoryLookup)
    ->Name("runtime/jit/alternating_memory_lookup")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);
BENCHMARK(bmBatchMemoryLookup)
    ->Name("runtime/jit/batch_memory_lookup")
    ->Arg(2)
    ->Arg(7);
BENCHMARK(bmDiskLookup)->Name("runtime/jit/disk_lookup");
BENCHMARK_CAPTURE(bmDiskFind, hit, true)->Name("runtime/jit/disk_find/hit")->UseRealTime();
BENCHMARK_CAPTURE(bmDiskFind, miss, false)->Name("runtime/jit/disk_find/miss")->UseRealTime();
BENCHMARK(bmLibraryLoad)->Name("runtime/jit/library_load");
BENCHMARK(bmPrepare)->Name("runtime/jit/prepare");
BENCHMARK(bmEmit)->Name("runtime/jit/emit");
BENCHMARK(bmPreparedExecute)
    ->Name("runtime/jit/prepared_execute")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);
BENCHMARK(bmCachedExecute)
    ->Name("runtime/jit/cached_execute")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);
