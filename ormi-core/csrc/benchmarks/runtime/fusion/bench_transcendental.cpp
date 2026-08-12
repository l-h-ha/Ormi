#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/kernel.h"
#include "backends/cpu/jit/math.h"
#include "core/runtime/fusion/expr.h"

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    Tensor tensor(std::int64_t elements, float value) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(result->data()),
            result->numel(),
            value
        );
        return result;
    }

    Tensor tensor(ormi::core::Shape shape, float value) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(result->data()),
            result->numel(),
            value
        );
        return result;
    }

    void preparedCosMul(benchmark::State& state, bool cosine) {
        const auto elements = state.range(0);
        const auto input = tensor(elements, 0.25f);
        const auto gradient = tensor(elements, 0.5f);
        const auto lhs = cosine
            ? ormi::core::ops::cos(input)
            : ormi::core::ops::neg(input);
        const auto output = ormi::core::ops::mul(lhs, gradient);
        const auto expression =
            ormi::core::runtime::fusion::build(*output).value();
        output->realize();
        auto setup = ormi::backends::cpu::jit::prepare(*output, expression);

        for (auto _ : state) {
            setup.run(*output);
            benchmark::ClobberMemory();
        }

        const auto actual = *static_cast<const float*>(output->data());
        const auto expected = cosine ? std::cos(0.25f) * 0.5f : -0.125f;
        if (std::abs(actual - expected) > 1.0e-6f) {
            state.SkipWithError("Prepared pointwise kernel produced an incorrect value.");
        }
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void preparedSinBias(benchmark::State& state) {
        constexpr std::int64_t width = 32;
        const auto batch = state.range(0);
        const auto input = tensor({batch, width}, 0.25f);
        const auto bias = tensor({width}, 0.5f);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(input, bias)
        );
        const auto expression =
            ormi::core::runtime::fusion::build(*output).value();
        output->realize();
        auto setup = ormi::backends::cpu::jit::prepare(*output, expression);

        for (auto _ : state) {
            setup.run(*output);
            benchmark::ClobberMemory();
        }

        const auto actual = *static_cast<const float*>(output->data());
        if (std::abs(actual - std::sin(0.75f)) > 1.0e-6f) {
            state.SkipWithError("Prepared pointwise kernel produced an incorrect value.");
        }
        state.SetItemsProcessed(state.iterations() * batch * width);
    }

    void preparedSinCosBias(benchmark::State& state) {
        constexpr std::int64_t width = 32;
        const auto batch = state.range(0);
        const auto input = tensor({batch, width}, 0.25f);
        const auto bias = tensor({width}, 0.5f);
        const auto mapped = ormi::core::ops::add(input, bias);
        const auto sine = ormi::core::ops::sin(mapped);
        const auto cosine = ormi::core::ops::cos(mapped);
        const std::array<ormi::core::Tensor*, 1> auxiliaries{cosine.get()};
        const std::unordered_set<const ormi::core::Tensor*> boundaries;
        const auto expression = ormi::core::runtime::fusion::build(
            *sine,
            auxiliaries,
            boundaries
        ).value();
        sine->realize();
        cosine->realize();
        auto setup = ormi::backends::cpu::jit::prepare(
            *sine,
            auxiliaries,
            expression
        );

        for (auto _ : state) {
            setup.run(*sine);
            benchmark::ClobberMemory();
        }

        const auto expected_sine = std::sin(0.75f);
        const auto expected_cosine = std::cos(0.75f);
        const auto actual_sine = *static_cast<const float*>(sine->data());
        const auto actual_cosine = *static_cast<const float*>(cosine->data());
        if (
            std::abs(actual_sine - expected_sine) > 1.0e-6f ||
            std::abs(actual_cosine - expected_cosine) > 1.0e-6f
        ) {
            state.SkipWithError(
                "Prepared paired sine and cosine produced an incorrect value."
            );
        }
        state.SetItemsProcessed(state.iterations() * batch * width);
    }

    void mathCallback(benchmark::State& state, std::size_t operation) {
        const auto math = ormi::backends::cpu::jit::math(
            ormi::backends::cpu::jit::MathABI::buffered
        );
        if (!math.functions) {
            state.SkipWithError("CPU vector math is unavailable.");
            return;
        }

        alignas(32) std::array<float, 8> input{};
        alignas(32) std::array<float, 8> output{};
        input.fill(0.25f);
        const auto function = static_cast<const ormi::backends::cpu::jit::Unary8*>(
            math.functions
        )[operation];
        const auto elements = static_cast<std::size_t>(state.range(0));

        for (auto _ : state) {
            for (std::size_t i = 0; i < elements; i += input.size()) {
                function(output.data(), input.data());
            }
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void sizes(benchmark::internal::Benchmark* benchmark) {
        benchmark
            ->Arg(1024)
            ->Arg(32768)
            ->Arg(524288)
            ->Arg(1048576);
    }
}

BENCHMARK_CAPTURE(preparedCosMul, cos_mul, true)
    ->Name("runtime/fusion/prepared_cos_mul")
    ->Apply(sizes);
BENCHMARK_CAPTURE(preparedCosMul, neg_mul_control, false)
    ->Name("runtime/fusion/prepared_neg_mul_control")
    ->Apply(sizes);
BENCHMARK(preparedSinBias)
    ->Name("runtime/fusion/prepared_sin_bias")
    ->Arg(1)
    ->Arg(32)
    ->Arg(1024);
BENCHMARK(preparedSinCosBias)
    ->Name("runtime/fusion/prepared_sin_cos_bias")
    ->Arg(1)
    ->Arg(32)
    ->Arg(1024);
BENCHMARK_CAPTURE(mathCallback, sin, 2)
    ->Name("runtime/fusion/math_callback/sin")
    ->Arg(32768);
BENCHMARK_CAPTURE(mathCallback, cos, 3)
    ->Name("runtime/fusion/math_callback/cos")
    ->Arg(32768);
