#include <cstdint>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/group.h"

namespace {
    using Tensor = ormi::core::Tensor;
    using Roots = std::vector<std::shared_ptr<Tensor>>;

    std::shared_ptr<Tensor> input() {
        return std::make_shared<Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float32
        );
    }

    Roots binary_chain(std::int64_t operations) {
        const auto source = input();
        auto result = source;
        for (std::int64_t operation = 0; operation < operations; ++operation) {
            result = ormi::core::ops::add(result, source);
        }
        return {result};
    }

    Roots unary_chain(std::int64_t operations) {
        auto result = input();
        for (std::int64_t operation = 0; operation < operations; ++operation) {
            result = ormi::core::ops::neg(result);
        }
        return {result};
    }

    Roots fanout(std::int64_t outputs) {
        const auto source = input();
        const auto parent = ormi::core::ops::add(source, source);
        Roots roots;
        roots.reserve(static_cast<std::size_t>(outputs));
        for (std::int64_t output = 0; output < outputs; ++output) {
            roots.push_back(ormi::core::ops::neg(parent));
        }
        return roots;
    }

    Roots many_groups(std::int64_t parents, std::int64_t outputs_per_parent) {
        const auto source = input();
        Roots roots;
        roots.reserve(static_cast<std::size_t>(parents * outputs_per_parent));

        for (std::int64_t parent = 0; parent < parents; ++parent) {
            const auto mapped = ormi::core::ops::add(source, source);
            for (std::int64_t output = 0; output < outputs_per_parent; ++output) {
                roots.push_back(ormi::core::ops::neg(mapped));
            }
        }
        return roots;
    }

    void measure(benchmark::State& state, const Roots& owners) {
        std::vector<Tensor*> roots;
        roots.reserve(owners.size());
        for (const auto& owner : owners) {
            roots.push_back(owner.get());
        }

        for (auto _ : state) {
            auto candidates = ormi::core::runtime::fusion::find_candidates(roots);
            benchmark::DoNotOptimize(candidates);
        }
    }

    void bmBinaryChain(benchmark::State& state) {
        measure(state, binary_chain(state.range(0)));
    }

    void bmUnaryChain(benchmark::State& state) {
        measure(state, unary_chain(state.range(0)));
    }

    void bmFanout(benchmark::State& state) {
        measure(state, fanout(state.range(0)));
    }

    void bmManyGroups(benchmark::State& state) {
        measure(state, many_groups(state.range(0), state.range(1)));
    }
}

#define ORMI_GROUP_SIZES(BENCHMARK_NAME) \
    BENCHMARK_NAME->Arg(1)->Arg(4)->Arg(16)->Arg(64)->Arg(256)

ORMI_GROUP_SIZES(
    BENCHMARK(bmBinaryChain)->Name("runtime/fusion/group/binary_chain")
);
ORMI_GROUP_SIZES(
    BENCHMARK(bmUnaryChain)->Name("runtime/fusion/group/unary_chain")
);
ORMI_GROUP_SIZES(
    BENCHMARK(bmFanout)->Name("runtime/fusion/group/fanout")
);
BENCHMARK(bmManyGroups)
    ->Name("runtime/fusion/group/many_groups")
    ->Args({1, 4})
    ->Args({4, 4})
    ->Args({16, 4})
    ->Args({64, 4})
    ->Args({4, 16});

#undef ORMI_GROUP_SIZES
