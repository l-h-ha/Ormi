#include <algorithm>
#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/reduction.h"

namespace {
    void bmSum(benchmark::State& state, std::vector<std::int64_t> axes) {
        const auto rows = state.range(0);
        const auto columns = state.range(1);
        ormi::core::Tensor input(
            ormi::core::Device::cpu(),
            {rows, columns},
            ormi::core::DType::float32
        );
        const auto output_shape = axes == std::vector<std::int64_t>{0}
            ? ormi::core::Shape{columns}
            : ormi::core::Shape{rows};
        ormi::core::Tensor output(
            ormi::core::Device::cpu(),
            output_shape,
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(input.data()),
            input.numel(),
            0.125f
        );

        ormi::backends::cpu::reduce(
            ormi::core::ops::ReduceOp::sum,
            output,
            input,
            axes,
            false
        );

        for (auto _ : state) {
            ormi::backends::cpu::reduce(
                ormi::core::ops::ReduceOp::sum,
                output,
                input,
                axes,
                false
            );
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(
            state.iterations() * rows * columns
        );
    }
}

BENCHMARK_CAPTURE(bmSum, first_axis, std::vector<std::int64_t>{0})
    ->Name("backend/cpu/reduction/first_axis_sum")
    ->Args({32, 32})
    ->Args({1024, 32});

BENCHMARK_CAPTURE(bmSum, last_axis, std::vector<std::int64_t>{1})
    ->Name("backend/cpu/reduction/last_axis_sum")
    ->Args({32, 32})
    ->Args({1024, 32});
