#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/plan.h>
#include <ormi/backend/api.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"
#include "core/runtime/device.h"
#include "core/runtime/fusion/expr.h"

namespace {
    enum class Reduction {
        sum,
        mean
    };

    enum class Timing {
        submit,
        host,
        complete
    };

    bool require_cuda(benchmark::State& state) {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess) {
            state.SkipWithError(cudaGetErrorString(status));
            return false;
        }
        if (device_count == 0) {
            state.SkipWithError("No CUDA device is available.");
            return false;
        }
        return true;
    }

    template <typename Run>
    void warmup_cuda(ormi::core::Device device, Run&& run) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
        do {
            run();
        } while (std::chrono::steady_clock::now() < deadline);
        ormi::core::runtime::device::synchronize(device);
    }

    template <typename Run>
    void measure_cuda(
        benchmark::State& state,
        ormi::core::Device device,
        Timing timing,
        Run&& run
    ) {
        if (timing == Timing::submit) {
            run();
            return;
        }

        if (timing == Timing::host) {
            ormi::core::runtime::device::synchronize(device);
            const auto start = std::chrono::steady_clock::now();
            run();
            const auto elapsed = std::chrono::steady_clock::now() - start;
            ormi::core::runtime::device::synchronize(device);
            state.SetIterationTime(
                std::chrono::duration<double>(elapsed).count()
            );
            return;
        }

        const auto start = std::chrono::steady_clock::now();
        run();
        ormi::core::runtime::device::synchronize(device);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        state.SetIterationTime(std::chrono::duration<double>(elapsed).count());
    }

    std::shared_ptr<ormi::core::Tensor> chain(
        const std::shared_ptr<ormi::core::Tensor>& input
    ) {
        return ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
    }

    std::shared_ptr<ormi::core::Tensor> arithmetic_chain(
        const std::shared_ptr<ormi::core::Tensor>& input
    ) {
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        return ormi::core::ops::mul(added, input);
    }

    std::shared_ptr<ormi::core::Tensor> input(
        std::int64_t elements,
        ormi::core::DType dtype,
        ormi::core::Device device
    ) {
        auto result = ormi::core::ops::full({elements}, 0.5f, device);
        if (dtype != result->dtype()) {
            result = ormi::core::ops::cast(result, dtype);
        }
        result->realize();
        return result;
    }

    void bmCUDAFusionFirstRealize(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        const auto input = ormi::core::ops::full({1024}, 0.5f, device);
        input->realize();

        for (auto _ : state) {
            auto output = ormi::core::ops::cos(ormi::core::ops::exp(chain(input)));
            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }
    }

    void bmCUDAFusion(
        benchmark::State& state,
        ormi::core::DType dtype,
        Timing timing
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(elements, dtype, device);
        auto warmup = chain(values);
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);
        warmup_cuda(device, [&] {
            auto output = chain(values);
            output->realize();
            benchmark::DoNotOptimize(output);
        });

        for (auto _ : state) {
            std::shared_ptr<ormi::core::Tensor> output;
            if (timing == Timing::submit) {
                state.PauseTiming();
            }
            output = chain(values);
            if (timing == Timing::submit) {
                state.ResumeTiming();
            }
            measure_cuda(state, device, timing, [&] {
                output->realize();
            });
            benchmark::DoNotOptimize(output);
            if (timing != Timing::submit) {
                output.reset();
                ormi::core::runtime::device::synchronize(device);
            }
        }

        ormi::core::runtime::device::synchronize(device);
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAFusionPlan(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            state.range(0),
            ormi::core::DType::float32,
            device
        );

        for (auto _ : state) {
            state.PauseTiming();
            auto output = chain(values);
            state.ResumeTiming();

            auto expression =
                ormi::core::runtime::fusion::build(*output);
            benchmark::DoNotOptimize(expression);

            state.PauseTiming();
            expression.reset();
            output.reset();
            state.ResumeTiming();
        }
    }

    void bmCUDAFusionPrepareState(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        const auto graph = chain(values);
        const auto expression =
            ormi::core::runtime::fusion::build(*graph);
        if (!expression) {
            state.SkipWithError("Cannot build CUDA fusion expression.");
            return;
        }

        auto output = ormi::core::ops::full(
            {elements},
            0.0f,
            device
        );
        output->realize();
        auto warmup = ormi::core::runtime::backend(
            device
        ).fusion.prepare(*output, {}, *expression);
        if (!warmup) {
            state.SkipWithError("Cannot prepare CUDA fusion execution.");
            return;
        }

        for (auto _ : state) {
            auto setup = ormi::core::runtime::backend(
                device
            ).fusion.prepare(*output, {}, *expression);
            benchmark::DoNotOptimize(setup.buffers());
        }
    }

    void bmCUDAFusionExecute(
        benchmark::State& state,
        Timing timing
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        const auto graph = chain(values);
        const auto expression =
            ormi::core::runtime::fusion::build(*graph);
        if (!expression) {
            state.SkipWithError("Cannot build CUDA fusion expression.");
            return;
        }

        auto output = ormi::core::ops::full(
            {elements},
            0.0f,
            device
        );
        output->realize();
        ormi::core::runtime::backend(device).fusion.execute(
            *output,
            {},
            *expression
        );
        ormi::core::runtime::device::synchronize(device);
        warmup_cuda(device, [&] {
            ormi::core::runtime::backend(device).fusion.execute(
                *output,
                {},
                *expression
            );
        });

        for (auto _ : state) {
            measure_cuda(state, device, timing, [&] {
                ormi::core::runtime::backend(device).fusion.execute(
                    *output,
                    {},
                    *expression
                );
            });
            benchmark::DoNotOptimize(output);
        }

        ormi::core::runtime::device::synchronize(device);
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAFusionPrepared(
        benchmark::State& state,
        Timing timing
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        const auto graph = chain(values);
        const auto expression =
            ormi::core::runtime::fusion::build(*graph);
        if (!expression) {
            state.SkipWithError("Cannot build CUDA fusion expression.");
            return;
        }

        auto output = ormi::core::ops::full(
            {elements},
            0.0f,
            device
        );
        output->realize();
        auto setup = ormi::core::runtime::backend(device).fusion.prepare(
            *output,
            {},
            *expression
        );
        if (!setup) {
            state.SkipWithError("Cannot prepare CUDA fusion execution.");
            return;
        }

        warmup_cuda(device, [&] {
            setup.run(*output);
        });

        for (auto _ : state) {
            measure_cuda(state, device, timing, [&] {
                setup.run(*output);
            });
            benchmark::DoNotOptimize(output);
        }

        ormi::core::runtime::device::synchronize(device);
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAFusionPreparedFresh(
        benchmark::State& state,
        Timing timing
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        const auto graph = chain(values);
        const auto expression =
            ormi::core::runtime::fusion::build(*graph);
        if (!expression) {
            state.SkipWithError("Cannot build CUDA fusion expression.");
            return;
        }

        ormi::core::Tensor initial(
            device,
            {elements},
            ormi::core::DType::float32
        );
        auto setup = ormi::core::runtime::backend(device).fusion.prepare(
            initial,
            {},
            *expression
        );
        if (!setup) {
            state.SkipWithError("Cannot prepare CUDA fusion execution.");
            return;
        }

        warmup_cuda(device, [&] {
            ormi::core::Tensor output(
                device,
                {elements},
                ormi::core::DType::float32
            );
            setup.run(output);
        });

        for (auto _ : state) {
            std::optional<ormi::core::Tensor> output;
            measure_cuda(state, device, timing, [&] {
                output.emplace(
                    device,
                    ormi::core::Shape{elements},
                    ormi::core::DType::float32
                );
                setup.run(*output);
            });
            benchmark::DoNotOptimize(output);
            if (timing != Timing::submit) {
                output.reset();
                ormi::core::runtime::device::synchronize(device);
            }
        }

        ormi::core::runtime::device::synchronize(device);
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAFusionReplay(
        benchmark::State& state,
        Timing timing
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        const auto output = chain(values);
        ormi::core::Plan plan({output});
        plan.run();
        ormi::core::runtime::device::synchronize(device);
        warmup_cuda(device, [&] {
            plan.run();
        });

        for (auto _ : state) {
            measure_cuda(state, device, timing, [&] {
                plan.run();
            });
            benchmark::DoNotOptimize(output);
        }

        ormi::core::runtime::device::synchronize(device);
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAFusionRecorded(
        benchmark::State& state,
        Timing timing
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        const auto output = chain(values);
        ormi::core::Plan plan({output});
        plan.run();
        ormi::core::runtime::device::synchronize(device);
        auto replay = plan.record();
        warmup_cuda(device, [&] {
            replay.run();
        });

        for (auto _ : state) {
            measure_cuda(state, device, timing, [&] {
                replay.run();
            });
            benchmark::DoNotOptimize(output);
        }

        ormi::core::runtime::device::synchronize(device);
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDASiblingOutputs(benchmark::State& state, bool grouped) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        const auto mapped = ormi::core::ops::add(values, values);
        const auto negative = ormi::core::ops::neg(mapped);
        const auto exponential = ormi::core::ops::exp(mapped);
        std::vector<ormi::core::Plan> plans;

        if (grouped) {
            plans.emplace_back(
                std::vector<std::shared_ptr<ormi::core::Tensor>>{
                    negative,
                    exponential
                }
            );
        } else {
            plans.emplace_back(
                std::vector<std::shared_ptr<ormi::core::Tensor>>{negative}
            );
            plans.emplace_back(
                std::vector<std::shared_ptr<ormi::core::Tensor>>{exponential}
            );
        }

        for (auto& plan : plans) {
            plan.run();
        }
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            for (auto& plan : plans) {
                plan.run();
            }
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(negative);
            benchmark::DoNotOptimize(exponential);
        }

        std::size_t steps = 0;
        for (const auto& plan : plans) {
            steps += plan.steps();
        }
        state.counters["steps"] = static_cast<double>(steps);
        state.SetItemsProcessed(state.iterations() * elements * 2);
    }

    void bmCUDAFusionMean(
        benchmark::State& state,
        bool prepared
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        const auto graph = ormi::core::ops::mean(chain(values));
        const auto expression =
            ormi::core::runtime::fusion::build(*graph);
        if (!expression) {
            state.SkipWithError("Cannot build CUDA mean fusion expression.");
            return;
        }

        auto output = ormi::core::ops::full(
            {},
            0.0f,
            device
        );
        output->realize();
        std::optional<ormi::core::Plan> plan;
        if (prepared) {
            plan.emplace(std::vector<std::shared_ptr<ormi::core::Tensor>>{
                graph
            });
            plan->run();
        } else {
            ormi::core::runtime::backend(
                device
            ).fusion.execute(*output, {}, *expression);
        }
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            if (prepared) {
                plan->run();
            } else {
                ormi::core::runtime::backend(
                    device
                ).fusion.execute(*output, {}, *expression);
            }
            benchmark::DoNotOptimize(output);
        }

        ormi::core::runtime::device::synchronize(device);
        if (plan) {
            state.counters["plan_buffers"] = static_cast<double>(
                plan->storage_buffers()
            );
            state.counters["plan_bytes"] = static_cast<double>(
                plan->storage_bytes()
            );
        }
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAUnfused(benchmark::State& state, ormi::core::DType dtype) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(elements, dtype, device);

        for (auto _ : state) {
            state.PauseTiming();
            auto square = ormi::core::ops::mul(values, values);
            auto sum = ormi::core::ops::add(square, values);
            auto output = ormi::core::ops::sin(sum);
            state.ResumeTiming();

            square->realize();
            sum->realize();
            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    std::shared_ptr<ormi::core::Tensor> reduce(
        const std::shared_ptr<ormi::core::Tensor>& input,
        Reduction reduction
    ) {
        return reduction == Reduction::sum
            ? ormi::core::ops::sum(input)
            : ormi::core::ops::mean(input);
    }

    void bmCUDAMapReduction(
        benchmark::State& state,
        Reduction reduction,
        bool force_materialized
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(elements, ormi::core::DType::float32, device);
        auto warmup_mapped = chain(values);
        auto warmup = reduce(warmup_mapped, reduction);
        if (force_materialized) {
            warmup_mapped->realize();
        }
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto mapped = chain(values);
            auto output = reduce(mapped, reduction);
            state.ResumeTiming();

            if (force_materialized) {
                mapped->realize();
            }
            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAArithmeticGlobalSum(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(
            elements,
            ormi::core::DType::float32,
            device
        );
        auto build = [&] {
            return ormi::core::ops::sum(arithmetic_chain(values));
        };

        auto warmup = build();
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto output = build();
            state.ResumeTiming();

            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAPartialMapSum(benchmark::State& state, bool force_materialized) {
        if (!require_cuda(state)) {
            return;
        }

        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto elements = rows * columns;
        const auto device = ormi::core::Device::from("cuda");
        const auto values = ormi::core::ops::full({rows, columns}, 0.5f, device);
        values->realize();

        auto warmup_mapped = chain(values);
        auto warmup = ormi::core::ops::sum(warmup_mapped, {1});
        if (force_materialized) {
            warmup_mapped->realize();
        }
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto mapped = chain(values);
            auto output = ormi::core::ops::sum(mapped, {1});
            state.ResumeTiming();

            if (force_materialized) {
                mapped->realize();
            }
            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDARank3Reduction(
        benchmark::State& state,
        bool mixed,
        bool partial
    ) {
        if (!require_cuda(state)) {
            return;
        }

        const auto dim0 = state.range(0);
        const auto dim1 = state.range(1);
        const auto dim2 = state.range(2);
        const auto elements = dim0 * dim1 * dim2;
        const auto device = ormi::core::Device::from("cuda");
        const auto base = ormi::core::ops::full(
            {dim1, dim2, dim0},
            0.5f,
            device
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto contiguous = mixed
            ? ormi::core::ops::full(
                {dim0, dim1, dim2},
                0.25f,
                device
            )
            : nullptr;
        base->realize();
        if (contiguous) {
            contiguous->realize();
        }

        auto build = [&] {
            std::shared_ptr<ormi::core::Tensor> mapped;
            if (mixed) {
                const auto square =
                    ormi::core::ops::mul(permuted, permuted);
                mapped = ormi::core::ops::mul(
                    ormi::core::ops::add(square, contiguous),
                    permuted
                );
            } else {
                mapped = arithmetic_chain(permuted);
            }
            return partial
                ? ormi::core::ops::sum(mapped, {1})
                : ormi::core::ops::sum(mapped);
        };

        auto warmup = build();
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto output = build();
            state.ResumeTiming();

            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAFirstAxisSum(benchmark::State& state) {
        if (!require_cuda(state)) {
            return;
        }

        const auto rows = state.range(0);
        const auto columns = state.range(1);
        const auto elements = rows * columns;
        const auto device = ormi::core::Device::from("cuda");
        const auto values =
            ormi::core::ops::full({rows, columns}, 0.5f, device);
        values->realize();

        auto build = [&] {
            return ormi::core::ops::sum(
                arithmetic_chain(values),
                {0}
            );
        };
        auto warmup = build();
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto output = build();
            state.ResumeTiming();

            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }

    void bmCUDAAutocastMapSum(benchmark::State& state, bool force_materialized) {
        if (!require_cuda(state)) {
            return;
        }

        const auto elements = state.range(0);
        const auto device = ormi::core::Device::from("cuda");
        const auto values = input(elements, ormi::core::DType::float16, device);

        auto build = [&] {
            const ormi::core::Autocast autocast;
            auto mapped = ormi::core::ops::sin(values);
            auto output = ormi::core::ops::sum(mapped);
            return std::pair{std::move(mapped), std::move(output)};
        };

        auto [warmup_mapped, warmup] = build();
        if (force_materialized) {
            warmup_mapped->realize();
        }
        warmup->realize();
        ormi::core::runtime::device::synchronize(device);

        for (auto _ : state) {
            state.PauseTiming();
            auto [mapped, output] = build();
            state.ResumeTiming();

            if (force_materialized) {
                mapped->realize();
            }
            output->realize();
            ormi::core::runtime::device::synchronize(device);
            benchmark::DoNotOptimize(output);
        }

        state.SetItemsProcessed(state.iterations() * elements);
    }
}

BENCHMARK(bmCUDAFusionFirstRealize)
    ->Name("runtime/cuda_fusion/first_realize_complete")
    ->Iterations(1)
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmCUDAFusion,
    submit,
    ormi::core::DType::float32,
    Timing::submit
)
    ->Name("runtime/cuda_fusion/cached_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmCUDAFusion,
    host,
    ormi::core::DType::float32,
    Timing::host
)
    ->Name("runtime/cuda_fusion/cached_host_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(
    bmCUDAFusion,
    complete,
    ormi::core::DType::float32,
    Timing::complete
)
    ->Name("runtime/cuda_fusion/cached_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK(bmCUDAFusionPlan)
    ->Name("runtime/cuda_fusion/plan")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);

BENCHMARK(bmCUDAFusionPrepareState)
    ->Name("runtime/cuda_fusion/prepare_state")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576);

BENCHMARK_CAPTURE(bmCUDAFusionExecute, submit, Timing::submit)
    ->Name("runtime/cuda_fusion/execute_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusionExecute, host, Timing::host)
    ->Name("runtime/cuda_fusion/execute_host_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDAFusionExecute, complete, Timing::complete)
    ->Name("runtime/cuda_fusion/execute_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDAFusionPrepared, submit, Timing::submit)
    ->Name("runtime/cuda_fusion/prepared_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusionPrepared, host, Timing::host)
    ->Name("runtime/cuda_fusion/prepared_host_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDAFusionPrepared, complete, Timing::complete)
    ->Name("runtime/cuda_fusion/prepared_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDAFusionPreparedFresh, submit, Timing::submit)
    ->Name("runtime/cuda_fusion/prepared_fresh_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusionPreparedFresh, host, Timing::host)
    ->Name("runtime/cuda_fusion/prepared_fresh_host_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(
    bmCUDAFusionPreparedFresh,
    complete,
    Timing::complete
)
    ->Name("runtime/cuda_fusion/prepared_fresh_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDAFusionReplay, submit, Timing::submit)
    ->Name("runtime/cuda_fusion/plan_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusionReplay, host, Timing::host)
    ->Name("runtime/cuda_fusion/plan_host_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDAFusionReplay, complete, Timing::complete)
    ->Name("runtime/cuda_fusion/plan_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDAFusionRecorded, submit, Timing::submit)
    ->Name("runtime/cuda_fusion/recorded_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusionRecorded, complete, Timing::complete)
    ->Name("runtime/cuda_fusion/recorded_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDASiblingOutputs, independent, false)
    ->Name("runtime/cuda_fusion/independent_outputs_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDASiblingOutputs, grouped, true)
    ->Name("runtime/cuda_fusion/grouped_outputs_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusionMean, execute, false)
    ->Name("runtime/cuda_fusion/mean_execute_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAFusionMean, plan, true)
    ->Name("runtime/cuda_fusion/mean_plan_submit")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAUnfused, float32, ormi::core::DType::float32)
    ->Name("runtime/cuda_fusion/unfused_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmCUDAFusion,
    float16,
    ormi::core::DType::float16,
    Timing::complete
)
    ->Name("runtime/cuda_fusion/float16_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseManualTime();

BENCHMARK_CAPTURE(bmCUDAUnfused, float16, ormi::core::DType::float16)
    ->Name("runtime/cuda_fusion/float16_unfused_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAMapReduction, sum_fused, Reduction::sum, false)
    ->Name("runtime/cuda_fusion/map_sum_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAMapReduction, sum_materialized, Reduction::sum, true)
    ->Name("runtime/cuda_fusion/materialized_map_sum_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAMapReduction, mean_fused, Reduction::mean, false)
    ->Name("runtime/cuda_fusion/map_mean_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAMapReduction, mean_materialized, Reduction::mean, true)
    ->Name("runtime/cuda_fusion/materialized_map_mean_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK(bmCUDAArithmeticGlobalSum)
    ->Name("runtime/cuda_fusion/arithmetic_global_sum_complete")
    ->Arg(1024)
    ->Arg(65536)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAPartialMapSum, fused, false)
    ->Name("runtime/cuda_fusion/partial_map_sum_complete")
    ->Args({1, 1})
    ->Args({32, 32})
    ->Args({1024, 1024})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAPartialMapSum, materialized, true)
    ->Name("runtime/cuda_fusion/materialized_partial_map_sum_complete")
    ->Args({1, 1})
    ->Args({32, 32})
    ->Args({1024, 1024})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmCUDARank3Reduction,
    permuted_global,
    false,
    false
)
    ->Name("runtime/cuda_fusion/permuted_rank3_global_sum_complete")
    ->Args({64, 64, 64})
    ->Args({32, 128, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmCUDARank3Reduction,
    mixed_global,
    true,
    false
)
    ->Name("runtime/cuda_fusion/mixed_rank3_global_sum_complete")
    ->Args({64, 64, 64})
    ->Args({32, 128, 256})
    ->UseRealTime();

BENCHMARK_CAPTURE(
    bmCUDARank3Reduction,
    permuted_middle_axis,
    false,
    true
)
    ->Name("runtime/cuda_fusion/permuted_rank3_middle_axis_sum_complete")
    ->Args({64, 64, 64})
    ->Args({32, 128, 256})
    ->UseRealTime();

BENCHMARK(bmCUDAFirstAxisSum)
    ->Name("runtime/cuda_fusion/first_axis_sum_complete")
    ->Args({32, 32})
    ->Args({4096, 32})
    ->Args({64, 256})
    ->Args({4096, 256})
    ->Args({1024, 1024})
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAutocastMapSum, fused, false)
    ->Name("runtime/cuda_fusion/autocast_map_sum_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();

BENCHMARK_CAPTURE(bmCUDAAutocastMapSum, materialized, true)
    ->Name("runtime/cuda_fusion/materialized_autocast_map_sum_complete")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1048576)
    ->UseRealTime();
