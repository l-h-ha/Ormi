#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"

namespace {
    using Tensor = ormi::core::Tensor;
    using Graph = std::shared_ptr<Tensor> (*)(std::int64_t);

    std::shared_ptr<Tensor> input() {
        return std::make_shared<Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1},
            ormi::core::DType::float32
        );
    }

    std::shared_ptr<Tensor> unary_chain(std::int64_t operations) {
        auto result = input();
        for (std::int64_t operation = 0; operation < operations; ++operation) {
            result = ormi::core::ops::neg(result);
        }
        return result;
    }

    std::shared_ptr<Tensor> shared_chain(std::int64_t operations) {
        const auto shared = input();
        auto result = shared;
        for (std::int64_t operation = 0; operation < operations; ++operation) {
            result = ormi::core::ops::add(result, shared);
        }
        return result;
    }

    std::shared_ptr<Tensor> distinct_chain(std::int64_t operations) {
        auto result = input();
        for (std::int64_t operation = 0; operation < operations; ++operation) {
            result = ormi::core::ops::add(result, input());
        }
        return result;
    }

    std::shared_ptr<Tensor> balanced_tree(std::int64_t operations) {
        std::vector<std::shared_ptr<Tensor>> level;
        const auto leaves = operations + 1;
        level.reserve(static_cast<std::size_t>(leaves));
        for (std::int64_t leaf = 0; leaf < leaves; ++leaf) {
            level.push_back(input());
        }

        while (level.size() > 1) {
            std::vector<std::shared_ptr<Tensor>> next;
            next.reserve((level.size() + 1) / 2);
            std::size_t index = 0;
            for (; index + 1 < level.size(); index += 2) {
                next.push_back(ormi::core::ops::add(level[index], level[index + 1]));
            }
            if (index < level.size()) {
                next.push_back(std::move(level[index]));
            }
            level = std::move(next);
        }
        return level.front();
    }

    void bmBuild(benchmark::State& state, Graph graph) {
        const auto root = graph(state.range(0));

        for (auto _ : state) {
            auto result = ormi::core::runtime::fusion::build(*root);
            benchmark::DoNotOptimize(result);
        }
    }

    void bmReject(benchmark::State& state) {
        const auto root = ormi::core::ops::matmul(input(), input());

        for (auto _ : state) {
            auto result = ormi::core::runtime::fusion::build(*root);
            benchmark::DoNotOptimize(result);
        }
    }
}

#define ORMI_BUILD_SIZES(BENCHMARK_NAME) \
    BENCHMARK_NAME->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(64)->Arg(256)

ORMI_BUILD_SIZES(
    BENCHMARK_CAPTURE(bmBuild, unary, unary_chain)
        ->Name("runtime/fusion/build/unary")
);
ORMI_BUILD_SIZES(
    BENCHMARK_CAPTURE(bmBuild, shared, shared_chain)
        ->Name("runtime/fusion/build/shared")
);
ORMI_BUILD_SIZES(
    BENCHMARK_CAPTURE(bmBuild, distinct, distinct_chain)
        ->Name("runtime/fusion/build/distinct")
);
ORMI_BUILD_SIZES(
    BENCHMARK_CAPTURE(bmBuild, balanced, balanced_tree)
        ->Name("runtime/fusion/build/balanced")
);
BENCHMARK(bmReject)->Name("runtime/fusion/build/reject");

#undef ORMI_BUILD_SIZES
