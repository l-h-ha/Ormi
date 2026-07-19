#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

namespace {
    std::shared_ptr<ormi::core::Tensor> scalarTensor() {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float32
        );
    }

    void bmSquareGradientConstruct(benchmark::State& state) {
        const auto x = scalarTensor();
        const auto y = ormi::core::ops::mul(x, x);

        for (auto _ : state) {
            auto result = ormi::core::grad(y, x);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmFifthDerivativeConstruct(benchmark::State& state) {
        const auto x = scalarTensor();
        const auto x2 = ormi::core::ops::mul(x, x);
        const auto x3 = ormi::core::ops::mul(x2, x);
        const auto x4 = ormi::core::ops::mul(x3, x);
        const auto x5 = ormi::core::ops::mul(x4, x);

        for (auto _ : state) {
            auto result = ormi::core::diff(x5, x, 5);
            benchmark::DoNotOptimize(result);
        }
    }
}

BENCHMARK(bmSquareGradientConstruct)->Name("graph/grad_square_construct");
BENCHMARK(bmFifthDerivativeConstruct)->Name("graph/diff_fifth_construct");
