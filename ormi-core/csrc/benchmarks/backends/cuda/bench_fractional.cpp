#include <cstdint>
#include <memory>
#include <optional>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/fractional.h>
#include <ormi/core/plan.h>

#include "core/runtime/dispatch/device.h"

namespace {
    bool require_fractional_cuda(benchmark::State& state) {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);

        if (status != cudaSuccess) {
            state.SkipWithError(cudaGetErrorString(status));
            return false;
        }
        if (devices == 0) {
            state.SkipWithError("No CUDA device is available.");
            return false;
        }
        return true;
    }

    void bmCUDAFractionalPrepare(
        benchmark::State& state,
        bool higher_order
    ) {
        if (!require_fractional_cuda(state)) {
            return;
        }

        const auto lines = state.range(0);
        const auto history = state.range(1);
        const auto device = ormi::core::Device::from("cuda");
        const auto input = ormi::core::ops::full(
            {lines, history},
            0.5f,
            device
        );
        input->realize();
        std::shared_ptr<ormi::core::Tensor> initial_derivative;
        if (higher_order) {
            initial_derivative = ormi::core::ops::full(
                {lines},
                0.25f,
                device
            );
            initial_derivative->realize();
        }

        ormi::core::runtime::device::synchronize(device);
        for (auto _ : state) {
            const auto result = ormi::core::ops::caputo(
                input,
                higher_order ? 1.5 : 0.5,
                0.01,
                -1,
                initial_derivative
            );
            {
                ormi::core::Plan plan({result});
                benchmark::DoNotOptimize(plan.storage_bytes());
            }
            ormi::core::runtime::device::synchronize(device);
        }
    }

    void bmCUDAFractional(
        benchmark::State& state,
        bool recorded,
        bool adjoint,
        bool higher_order
    ) {
        if (!require_fractional_cuda(state)) {
            return;
        }

        const auto lines = state.range(0);
        const auto history = state.range(1);
        const auto device = ormi::core::Device::from("cuda");
        const auto input = ormi::core::ops::full(
            {lines, history},
            0.5f,
            device
        );
        std::shared_ptr<ormi::core::Tensor> initial_derivative;
        if (higher_order) {
            initial_derivative = ormi::core::ops::full(
                {lines},
                0.25f,
                device
            );
        }
        auto result = ormi::core::ops::caputo(
            input,
            higher_order ? 1.5 : 0.5,
            0.01,
            -1,
            initial_derivative
        );

        if (adjoint) {
            const auto seed = ormi::core::ops::full(
                {lines, history},
                0.25f,
                device
            );
            result = ormi::core::grad(result, input, seed);
        }
        ormi::core::Plan plan({result});
        std::optional<ormi::core::Plan::Replay> replay;

        plan.run();
        ormi::core::runtime::device::synchronize(device);

        if (recorded) {
            replay.emplace(plan.record());
            replay->run();
            ormi::core::runtime::device::synchronize(device);
        }

        for (auto _ : state) {
            if (recorded) {
                replay->run();
            } else {
                plan.run();
            }
            ormi::core::runtime::device::synchronize(device);
        }

        const auto forward_terms = history > 1
            ? static_cast<double>(history) *
                static_cast<double>(history - 1) / 2.0
            : 0.0;
        const auto adjoint_boundary_terms = history > 1
            ? static_cast<double>(history - 1)
            : 0.0;
        const auto terms_per_line = adjoint
            ? forward_terms + adjoint_boundary_terms
            : forward_terms;
        state.counters["plan_buffers"] = static_cast<double>(
            plan.storage_buffers()
        );
        state.counters["plan_bytes"] = static_cast<double>(
            plan.storage_bytes()
        );
        state.counters[higher_order ? "l2_terms" : "l1_terms"] =
            benchmark::Counter(
            static_cast<double>(state.iterations()) *
                static_cast<double>(lines) *
                terms_per_line,
            benchmark::Counter::kIsRate
        );
    }
}

BENCHMARK_CAPTURE(bmCUDAFractionalPrepare, l1, false)
    ->Name("graph/cuda_caputo/prepare_complete")
    ->Args({1, 64})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractionalPrepare, l2, true)
    ->Name("graph/cuda_caputo_l2/prepare_complete")
    ->Args({1, 64})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractional, plan, false, false, false)
    ->Name("graph/cuda_caputo/plan_complete")
    ->Args({1, 32})
    ->Args({1, 64})
    ->Args({1, 128})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->Args({32, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractional, recorded, true, false, false)
    ->Name("graph/cuda_caputo/recorded_complete")
    ->Args({1, 32})
    ->Args({1, 64})
    ->Args({1, 128})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->Args({32, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractional, adjoint_plan, false, true, false)
    ->Name("graph/cuda_caputo_adjoint/plan_complete")
    ->Args({1, 32})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->Args({32, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractional, adjoint_recorded, true, true, false)
    ->Name("graph/cuda_caputo_adjoint/recorded_complete")
    ->Args({1, 32})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->Args({32, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractional, l2_plan, false, false, true)
    ->Name("graph/cuda_caputo_l2/plan_complete")
    ->Args({1, 32})
    ->Args({1, 64})
    ->Args({1, 128})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->Args({32, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractional, l2_recorded, true, false, true)
    ->Name("graph/cuda_caputo_l2/recorded_complete")
    ->Args({1, 32})
    ->Args({1, 64})
    ->Args({1, 128})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->Args({32, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractional, l2_adjoint_plan, false, true, true)
    ->Name("graph/cuda_caputo_l2_adjoint/plan_complete")
    ->Args({1, 32})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->Args({32, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFractional, l2_adjoint_recorded, true, true, true)
    ->Name("graph/cuda_caputo_l2_adjoint/recorded_complete")
    ->Args({1, 32})
    ->Args({1, 256})
    ->Args({1, 2048})
    ->Args({1, 8192})
    ->Args({32, 256})
    ->UseRealTime();
