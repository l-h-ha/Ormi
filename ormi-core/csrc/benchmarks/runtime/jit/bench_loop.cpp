#include <cstddef>
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

#include "backends/cpu/jit/loop.h"
#include "core/runtime/fusion/expr.h"

namespace {
    enum class Layout {
        flat,
        row,
        column,
        expanded,
        transposed,
        global_reduction,
        last_axis_reduction,
        first_axis_reduction,
        separated_axes_reduction,
        conflicting_reduction
    };

    std::shared_ptr<ormi::core::Tensor> tensor(
        ormi::core::Shape shape,
        ormi::core::DType dtype
    ) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), std::move(shape), dtype
        );
    }

    void bmLoop(benchmark::State& state, Layout layout) {
        using namespace ormi::core;

        const auto dtype = state.range(0) == 16 ? DType::float16
            : state.range(0) == 32 ? DType::float32 : DType::float64;
        auto input = tensor({32, 64}, dtype);
        auto rhs = input;
        switch (layout) {
            case Layout::flat:
                input = tensor({2048}, dtype);
                rhs = input;
                break;
            case Layout::row:
                rhs = tensor({64}, dtype);
                break;
            case Layout::column:
                rhs = tensor({32, 1}, dtype);
                break;
            case Layout::expanded:
                rhs = ops::expand(tensor({32, 1}, dtype), {32, 64});
                break;
            case Layout::transposed:
            case Layout::conflicting_reduction:
                input = ops::transpose(tensor({64, 32}, dtype), {1, 0});
                break;
            case Layout::separated_axes_reduction:
                input = tensor({4, 32, 64}, dtype);
                rhs = tensor({64}, dtype);
                break;
            case Layout::global_reduction:
            case Layout::last_axis_reduction:
            case Layout::first_axis_reduction:
                break;
        }

        auto root = ops::mul(input, input);
        for (std::int64_t i = 0; i < state.range(1); ++i) {
            root = ops::add(root, rhs);
        }
        switch (layout) {
            case Layout::global_reduction:
            case Layout::conflicting_reduction:
                root = ops::sum(root);
                break;
            case Layout::last_axis_reduction:
                root = ops::sum(root, {1});
                break;
            case Layout::first_axis_reduction:
                root = ops::sum(root, {0});
                break;
            case Layout::separated_axes_reduction:
                root = ops::sum(root, {0, 2});
                break;
            default:
                break;
        }

        const auto expression = runtime::fusion::build(*root).value();
        for (auto _ : state) {
            auto loop = ormi::backends::cpu::jit::plan(expression);
            benchmark::DoNotOptimize(loop);
        }
        state.counters["nodes"] = static_cast<double>(expression.nodes.size());
        state.counters["inputs"] = static_cast<double>(expression.inputs.size());
    }

    void cases(benchmark::internal::Benchmark* benchmark) {
        benchmark->ArgsProduct({{16, 32, 64}, {4, 32}});
        benchmark->ArgNames({"bits", "adds"});
        benchmark->UseRealTime();
    }
}

BENCHMARK_CAPTURE(bmLoop, flat, Layout::flat)
    ->Name("runtime/jit/loop/flat")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, row, Layout::row)
    ->Name("runtime/jit/loop/row")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, column, Layout::column)
    ->Name("runtime/jit/loop/column")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, expanded, Layout::expanded)
    ->Name("runtime/jit/loop/expanded")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, transposed, Layout::transposed)
    ->Name("runtime/jit/loop/transposed")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, global_reduction, Layout::global_reduction)
    ->Name("runtime/jit/loop/global_reduction")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, last_axis_reduction, Layout::last_axis_reduction)
    ->Name("runtime/jit/loop/last_axis_reduction")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, first_axis_reduction, Layout::first_axis_reduction)
    ->Name("runtime/jit/loop/first_axis_reduction")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, separated_axes_reduction, Layout::separated_axes_reduction)
    ->Name("runtime/jit/loop/separated_axes_reduction")->Apply(cases);
BENCHMARK_CAPTURE(bmLoop, conflicting_reduction, Layout::conflicting_reduction)
    ->Name("runtime/jit/loop/conflicting_reduction")->Apply(cases);
