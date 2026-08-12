#include <cstdint>
#include <memory>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"
#include "core/runtime/fusion/key.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> graph(std::int64_t operations) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float32
        );
        auto output = input;
        for (std::int64_t operation = 0; operation < operations; ++operation) {
            output = ormi::core::ops::add(output, input);
        }
        return output;
    }

    void bmHash(benchmark::State& state) {
        const auto root = graph(state.range(0));
        const auto expression =
            ormi::core::runtime::fusion::build(*root).value();

        for (auto _ : state) {
            auto hash =
                ormi::core::runtime::fusion::Key::hash(expression);
            benchmark::DoNotOptimize(hash);
        }
    }

    void bmConstruct(benchmark::State& state) {
        const auto root = graph(state.range(0));
        const auto expression =
            ormi::core::runtime::fusion::build(*root).value();

        for (auto _ : state) {
            const ormi::core::runtime::fusion::Key key(expression);
            benchmark::DoNotOptimize(key.hash());
        }
    }

    void bmMatch(benchmark::State& state) {
        const auto first_root = graph(state.range(0));
        const auto second_root = graph(state.range(0));
        const auto first =
            ormi::core::runtime::fusion::build(*first_root).value();
        const auto second =
            ormi::core::runtime::fusion::build(*second_root).value();
        const ormi::core::runtime::fusion::Key key(first);

        for (auto _ : state) {
            benchmark::DoNotOptimize(key.matches(second));
        }
    }
}

#define ORMI_KEY_SIZES(BENCHMARK_NAME) \
    BENCHMARK_NAME->Arg(2)->Arg(8)->Arg(64)->Arg(256)

ORMI_KEY_SIZES(BENCHMARK(bmHash)->Name("runtime/fusion/key_hash"));
ORMI_KEY_SIZES(BENCHMARK(bmConstruct)->Name("runtime/fusion/key_construct"));
ORMI_KEY_SIZES(BENCHMARK(bmMatch)->Name("runtime/fusion/key_match"));

#undef ORMI_KEY_SIZES
