#include <cstdint>
#include <memory>
#include <utility>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/codegen.h"
#include "backends/cpu/jit/loop.h"
#include "core/runtime/fusion/expr.h"

namespace {
    enum class Layout {
        scalar,
        contiguous,
        last_axis,
        separated_axes,
        tiled_output,
        tiled_global,
        strided_global
    };

    std::shared_ptr<ormi::core::Tensor> tensor(
        ormi::core::Shape shape, ormi::core::DType dtype
    ) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), std::move(shape), dtype
        );
    }

    void bmReductionCodegen(benchmark::State& state, Layout layout) {
        using namespace ormi::core;

        const auto dtype = state.range(0) == 16 ? DType::float16
            : state.range(0) == 32 ? DType::float32 : DType::float64;
        auto input = tensor({65}, dtype);
        auto rhs = input;
        switch (layout) {
            case Layout::scalar:
                input = tensor({7}, dtype);
                rhs = input;
                break;
            case Layout::last_axis:
                input = tensor({4, 65}, dtype);
                rhs = input;
                break;
            case Layout::separated_axes:
                input = tensor({4, 3, 65}, dtype);
                rhs = input;
                break;
            case Layout::tiled_output:
                input = tensor({65, 70}, dtype);
                rhs = input;
                break;
            case Layout::tiled_global:
            case Layout::strided_global:
                input = ops::transpose(tensor({70, 65}, dtype), {1, 0});
                rhs = layout == Layout::tiled_global
                    ? tensor({65, 70}, dtype) : input;
                break;
            case Layout::contiguous:
                break;
        }

        auto mapped = ops::add(ops::mul(input, input), rhs);
        auto root = layout == Layout::last_axis ? ops::sum(mapped, {1})
            : layout == Layout::separated_axes ? ops::sum(mapped, {0, 2})
            : layout == Layout::tiled_output ? ops::sum(mapped, {0})
            : ops::sum(mapped);
        const auto expression = runtime::fusion::build(*root).value();
        const auto loop = ormi::backends::cpu::jit::plan(expression);
        const auto bytes = ormi::backends::cpu::jit::emit(expression, loop).size();
        for (auto _ : state) {
            const auto source = ormi::backends::cpu::jit::emit(expression, loop);
            benchmark::DoNotOptimize(source.data());
            benchmark::DoNotOptimize(source.size());
        }
        state.counters["source_bytes"] = static_cast<double>(bytes);
        state.counters["nodes"] = static_cast<double>(expression.nodes.size());
    }

    void cases(benchmark::internal::Benchmark* benchmark) {
        benchmark->Arg(16)->Arg(32)->Arg(64)->ArgName("bits")->UseRealTime();
    }
}

BENCHMARK_CAPTURE(bmReductionCodegen, scalar, Layout::scalar)
    ->Name("runtime/jit/reduction/codegen/scalar")->Apply(cases);
BENCHMARK_CAPTURE(bmReductionCodegen, contiguous, Layout::contiguous)
    ->Name("runtime/jit/reduction/codegen/contiguous")->Apply(cases);
BENCHMARK_CAPTURE(bmReductionCodegen, last_axis, Layout::last_axis)
    ->Name("runtime/jit/reduction/codegen/last_axis")->Apply(cases);
BENCHMARK_CAPTURE(bmReductionCodegen, separated_axes, Layout::separated_axes)
    ->Name("runtime/jit/reduction/codegen/separated_axes")->Apply(cases);
BENCHMARK_CAPTURE(bmReductionCodegen, tiled_output, Layout::tiled_output)
    ->Name("runtime/jit/reduction/codegen/tiled_output")->Apply(cases);
BENCHMARK_CAPTURE(bmReductionCodegen, tiled_global, Layout::tiled_global)
    ->Name("runtime/jit/reduction/codegen/tiled_global")->Apply(cases);
BENCHMARK_CAPTURE(bmReductionCodegen, strided_global, Layout::strided_global)
    ->Name("runtime/jit/reduction/codegen/strided_global")->Apply(cases);
