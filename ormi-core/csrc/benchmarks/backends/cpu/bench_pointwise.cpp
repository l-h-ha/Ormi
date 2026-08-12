#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/pointwise.h"

namespace {
    void bmBinary(
        benchmark::State& state,
        ormi::core::ops::BinaryOp op
    ) {
        const auto elements = state.range(0);
        ormi::core::Tensor lhs(
            ormi::core::Device::cpu(),
            {elements},
            ormi::core::DType::float32
        );
        ormi::core::Tensor rhs(
            ormi::core::Device::cpu(),
            {elements},
            ormi::core::DType::float32
        );
        ormi::core::Tensor output(
            ormi::core::Device::cpu(),
            {elements},
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(lhs.data()),
            lhs.numel(),
            0.25f
        );
        std::fill_n(
            static_cast<float*>(rhs.data()),
            rhs.numel(),
            0.5f
        );

        ormi::backends::cpu::pointwise_binary(op, output, lhs, rhs);
        for (auto _ : state) {
            ormi::backends::cpu::pointwise_binary(op, output, lhs, rhs);
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }
}

#define ORMI_CPU_POINTWISE_SIZES(BENCHMARK_NAME) \
    BENCHMARK_NAME->Arg(1)->Arg(1024)->Arg(32 * 1024)

ORMI_CPU_POINTWISE_SIZES(
    BENCHMARK_CAPTURE(bmBinary, add, ormi::core::ops::BinaryOp::add)
        ->Name("backend/cpu/pointwise/add")
);
ORMI_CPU_POINTWISE_SIZES(
    BENCHMARK_CAPTURE(bmBinary, sub, ormi::core::ops::BinaryOp::sub)
        ->Name("backend/cpu/pointwise/sub")
);
ORMI_CPU_POINTWISE_SIZES(
    BENCHMARK_CAPTURE(bmBinary, mul, ormi::core::ops::BinaryOp::mul)
        ->Name("backend/cpu/pointwise/mul")
);
ORMI_CPU_POINTWISE_SIZES(
    BENCHMARK_CAPTURE(bmBinary, div, ormi::core::ops::BinaryOp::div)
        ->Name("backend/cpu/pointwise/div")
);

#undef ORMI_CPU_POINTWISE_SIZES
