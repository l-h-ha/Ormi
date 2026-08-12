#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/blas/openblas.h"
#include "backends/cpu/linalg.h"

#if defined(ORMI_HAS_AVX2_KERNELS)
#include "backends/cpu/matmul/avx2.h"
#include "platform/platform.h"
#endif

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    enum class Layout {
        contiguous,
        transposed
    };

    Tensor matrix(std::int64_t rows, std::int64_t columns) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{rows, columns},
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(result->data()),
            result->numel(),
            0.125f
        );
        return result;
    }

    Tensor operand(
        std::int64_t rows,
        std::int64_t columns,
        Layout layout
    ) {
        if (layout == Layout::contiguous) {
            return matrix(rows, columns);
        }

        return ormi::core::ops::transpose(
            matrix(columns, rows),
            {1, 0}
        );
    }

    void bmMatmul(
        benchmark::State& state,
        Layout lhs_layout,
        Layout rhs_layout
    ) {
        const auto rows = state.range(0);
        const auto inner = state.range(1);
        const auto columns = state.range(2);
        const auto lhs = operand(rows, inner, lhs_layout);
        const auto rhs = operand(inner, columns, rhs_layout);
        ormi::core::Tensor output(
            ormi::core::Device::cpu(),
            {rows, columns},
            ormi::core::DType::float32
        );

        ormi::backends::cpu::matmul(output, *lhs, *rhs);

        for (auto _ : state) {
            ormi::backends::cpu::matmul(output, *lhs, *rhs);
            benchmark::ClobberMemory();
        }

        const auto operations =
            2.0 * static_cast<double>(rows) *
            static_cast<double>(inner) *
            static_cast<double>(columns);
        state.counters["FLOP/s"] = benchmark::Counter(
            operations * static_cast<double>(state.iterations()),
            benchmark::Counter::kIsRate
        );
    }

    void bmOpenBLAS(
        benchmark::State& state,
        Layout lhs_layout,
        Layout rhs_layout
    ) {
        const auto rows = state.range(0);
        const auto inner = state.range(1);
        const auto columns = state.range(2);
        const auto lhs = operand(rows, inner, lhs_layout);
        const auto rhs = operand(inner, columns, rhs_layout);
        ormi::core::Tensor output(
            ormi::core::Device::cpu(),
            {rows, columns},
            ormi::core::DType::float32
        );
        const auto lhs_strides = lhs->strides();
        const auto rhs_strides = rhs->strides();
        const auto lhs_matrix = ormi::backends::cpu::blas::Matrix{
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(inner),
            lhs_strides[0],
            lhs_strides[1]
        };
        const auto rhs_matrix = ormi::backends::cpu::blas::Matrix{
            static_cast<std::size_t>(inner),
            static_cast<std::size_t>(columns),
            rhs_strides[0],
            rhs_strides[1]
        };
        auto* out = static_cast<float*>(output.data());
        const auto* lhs_data = static_cast<const float*>(lhs->data());
        const auto* rhs_data = static_cast<const float*>(rhs->data());

        if (!ormi::backends::cpu::blas::gemm(
                out,
                lhs_data,
                lhs_matrix,
                rhs_data,
                rhs_matrix
            )) {
            state.SkipWithError("OpenBLAS is unavailable for this layout.");
            return;
        }

        for (auto _ : state) {
            ormi::backends::cpu::blas::gemm(
                out,
                lhs_data,
                lhs_matrix,
                rhs_data,
                rhs_matrix
            );
            benchmark::ClobberMemory();
        }

        const auto operations =
            2.0 * static_cast<double>(rows) *
            static_cast<double>(inner) *
            static_cast<double>(columns);
        state.counters["FLOP/s"] = benchmark::Counter(
            operations * static_cast<double>(state.iterations()),
            benchmark::Counter::kIsRate
        );
    }

#if defined(ORMI_HAS_AVX2_KERNELS)
    void bmAVX2(
        benchmark::State& state,
        Layout lhs_layout
    ) {
        if (!ormi::platform::supports_fma()) {
            state.SkipWithError("The processor does not support AVX2 and FMA.");
            return;
        }

        const auto rows = static_cast<std::size_t>(state.range(0));
        const auto inner = static_cast<std::size_t>(state.range(1));
        const auto columns = static_cast<std::size_t>(state.range(2));
        const auto lhs = operand(
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(inner),
            lhs_layout
        );
        const auto rhs = matrix(
            static_cast<std::int64_t>(inner),
            static_cast<std::int64_t>(columns)
        );
        ormi::core::Tensor output(
            ormi::core::Device::cpu(),
            {
                static_cast<std::int64_t>(rows),
                static_cast<std::int64_t>(columns)
            },
            ormi::core::DType::float32
        );
        auto* out = static_cast<float*>(output.data());
        const auto* lhs_data = static_cast<const float*>(lhs->data());
        const auto* rhs_data = static_cast<const float*>(rhs->data());
        const auto work = rows * inner * columns;

        auto run = [&]() {
            if (lhs_layout == Layout::transposed) {
                ormi::backends::cpu::avx2::parallel_column_lhs_gemm(
                    out,
                    lhs_data,
                    rhs_data,
                    rows,
                    inner,
                    columns,
                    static_cast<std::int64_t>(rows),
                    static_cast<std::int64_t>(columns),
                    work
                );
                return;
            }

            ormi::backends::cpu::avx2::parallel_row_gemm(
                out,
                lhs_data,
                rhs_data,
                rows,
                inner,
                columns,
                static_cast<std::int64_t>(inner),
                1,
                static_cast<std::int64_t>(columns),
                static_cast<std::int64_t>(columns),
                work
            );
        };

        run();
        for (auto _ : state) {
            run();
            benchmark::ClobberMemory();
        }

        const auto operations = 2.0 * static_cast<double>(work);
        state.counters["FLOP/s"] = benchmark::Counter(
            operations * static_cast<double>(state.iterations()),
            benchmark::Counter::kIsRate
        );
    }
#endif

    void bmMatmulShape(benchmark::State& state) {
        const ormi::core::Shape lhs{
            state.range(0),
            state.range(1)
        };
        const ormi::core::Shape rhs{
            state.range(1),
            state.range(2)
        };

        for (auto _ : state) {
            auto shape = ormi::core::ops::matmul_shape(lhs, rhs);
            benchmark::DoNotOptimize(shape);
        }
    }
}

BENCHMARK(bmMatmulShape)
    ->Name("backend/cpu/matmul/shape")
    ->Args({1, 16, 32})
    ->Args({1, 32, 1})
    ->Args({16, 1, 32});

BENCHMARK_CAPTURE(
    bmMatmul,
    forward_hidden,
    Layout::contiguous,
    Layout::contiguous
)->Name("backend/cpu/matmul/forward_hidden")
    ->Args({1, 16, 32})
    ->Args({32, 16, 32})
    ->Args({1024, 16, 32})
    ->Args({2048, 512, 512})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmMatmul,
    forward_output,
    Layout::contiguous,
    Layout::contiguous
)->Name("backend/cpu/matmul/forward_output")
    ->Args({1, 32, 1})
    ->Args({32, 32, 1})
    ->Args({1024, 32, 1})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmMatmul,
    backward_second_weight,
    Layout::transposed,
    Layout::contiguous
)->Name("backend/cpu/matmul/backward_second_weight")
    ->Args({32, 1, 1})
    ->Args({32, 32, 1})
    ->Args({32, 1024, 1})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmMatmul,
    backward_hidden,
    Layout::contiguous,
    Layout::transposed
)->Name("backend/cpu/matmul/backward_hidden")
    ->Args({1, 1, 32})
    ->Args({32, 1, 32})
    ->Args({1024, 1, 32})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmMatmul,
    backward_first_weight,
    Layout::transposed,
    Layout::contiguous
)->Name("backend/cpu/matmul/backward_first_weight")
    ->Args({16, 1, 32})
    ->Args({16, 32, 32})
    ->Args({16, 1024, 32})
    ->Args({512, 2048, 512})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmOpenBLAS,
    forward_hidden,
    Layout::contiguous,
    Layout::contiguous
)->Name("backend/cpu/openblas/forward_hidden")
    ->Args({1, 16, 32})
    ->Args({32, 16, 32})
    ->Args({1024, 16, 32})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmOpenBLAS,
    backward_first_weight,
    Layout::transposed,
    Layout::contiguous
)->Name("backend/cpu/openblas/backward_first_weight")
    ->Args({16, 1, 32})
    ->Args({16, 32, 32})
    ->Args({16, 1024, 32})
    ->UseRealTime();

#define ORMI_GEMM_SHAPES(benchmark) \
    benchmark \
        ->Args({1024, 64, 32}) \
        ->Args({64, 1024, 32}) \
        ->Args({128, 128, 128}) \
        ->Args({256, 256, 256}) \
        ->Args({512, 512, 512}) \
        ->Args({512, 2048, 512}) \
        ->Args({2048, 512, 512}) \
        ->Args({4096, 64, 64}) \
        ->UseRealTime()

ORMI_GEMM_SHAPES(
    BENCHMARK_CAPTURE(
        bmOpenBLAS,
        general,
        Layout::contiguous,
        Layout::contiguous
    )->Name("backend/cpu/openblas/general")
);

#if defined(ORMI_HAS_AVX2_KERNELS)
ORMI_GEMM_SHAPES(
    BENCHMARK_CAPTURE(
        bmAVX2,
        general,
        Layout::contiguous
    )->Name("backend/cpu/avx2/general")
);

BENCHMARK_CAPTURE(
    bmOpenBLAS,
    transposed_lhs,
    Layout::transposed,
    Layout::contiguous
)->Name("backend/cpu/openblas/transposed_lhs")
    ->Args({64, 1024, 32})
    ->Args({512, 2048, 512})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmAVX2,
    transposed_lhs,
    Layout::transposed
)->Name("backend/cpu/avx2/transposed_lhs")
    ->Args({64, 1024, 32})
    ->Args({512, 2048, 512})
    ->UseRealTime();
#endif

#undef ORMI_GEMM_SHAPES
