#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/executor.h"
#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/execute.h"
#include "core/runtime/fusion/expr.h"

namespace {
    namespace cpu = ormi::backends::cpu;
    namespace ops = ormi::core::ops;

    enum class Stage { pointers, kernel, executor, direct, prepared, plan };

    struct Invocation {
        const cpu::jit::Compiled& compiled;
        void* const* outputs;
        const void* const* inputs;
        std::size_t elements;
    };

    void invoke(void* opaque, std::size_t, std::size_t begin, std::size_t end) {
        const auto& call = *static_cast<const Invocation*>(opaque);
        call.compiled.function(
            call.outputs, call.inputs, call.elements, begin, end, call.compiled.math
        );
    }

    std::shared_ptr<ormi::core::Tensor> input(ormi::core::Shape shape, float offset) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), shape, ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(result->data());
        for (std::size_t i = 0; i < result->numel(); ++i) {
            values[i] = offset + static_cast<float>(i % 31) * 0.0078125f;
        }
        return result;
    }

    void bmFusedExecution(benchmark::State& state, Stage stage) {
        const auto elements = state.range(0);
        const auto width = state.range(1);
        const bool broadcast = state.range(2) != 0;
        const ormi::core::Shape shape{elements / width, width};
        const auto values = input(shape, -0.25f);
        const auto scale = input(broadcast ? ormi::core::Shape{width} : shape, 0.125f);
        const auto bias = input(broadcast ? ormi::core::Shape{width} : shape, -0.0625f);
        const auto factor = ops::full({}, 0.125f, values->device());
        const auto transformed = ops::add(ops::mul(values, scale), bias);
        const auto output = ops::add(
            ops::mul(transformed, ops::sub(values, bias)), ops::mul(values, factor)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output).value();
        const auto compiled = cpu::jit::compile(expression);
        ormi::core::Plan plan({output});
        plan.run();
        if (plan.steps() != 2) {
            state.SkipWithError("Expected scalar creation followed by one fused pointwise step.");
            return;
        }
        auto setup = cpu::jit::execution::prepare({}, expression, compiled);
        std::vector<const void*> inputs(expression.inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = expression.inputs[i]->data();
        }
        void* outputs[]{output->data()};
        Invocation call{*compiled, outputs, inputs.data(), expression.elements};

        switch (stage) {
            case Stage::pointers:
                for (auto _ : state) {
                    for (std::size_t i = 0; i < inputs.size(); ++i) {
                        inputs[i] = expression.inputs[i]->data();
                    }
                    outputs[0] = output->data();
                    benchmark::DoNotOptimize(inputs.data());
                    benchmark::DoNotOptimize(outputs[0]);
                    benchmark::ClobberMemory();
                }
                break;
            case Stage::kernel:
                for (auto _ : state) {
                    invoke(&call, 0, 0, expression.elements);
                    benchmark::ClobberMemory();
                }
                break;
            case Stage::executor:
                for (auto _ : state) {
                    cpu::executor::run(
                        expression.elements, compiled->unit, compiled->work, invoke, &call
                    );
                    benchmark::ClobberMemory();
                }
                break;
            case Stage::direct:
                for (auto _ : state) {
                    cpu::jit::execution::run(*output, {}, expression, *compiled);
                    benchmark::ClobberMemory();
                }
                break;
            case Stage::prepared:
                for (auto _ : state) {
                    setup.run(*output);
                    benchmark::ClobberMemory();
                }
                break;
            case Stage::plan:
                for (auto _ : state) {
                    plan.run();
                    benchmark::ClobberMemory();
                }
                break;
        }

        const auto* x = static_cast<const float*>(values->data());
        const auto* s = static_cast<const float*>(scale->data());
        const auto* b = static_cast<const float*>(bias->data());
        const auto* actual = static_cast<const float*>(output->data());
        for (std::size_t i = 0; i < expression.elements; ++i) {
            const auto index = broadcast ? i % static_cast<std::size_t>(width) : i;
            const auto expected = (x[i] * s[index] + b[index]) * (x[i] - b[index]) + x[i] * 0.125f;
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected) > 1e-6f) {
                state.SkipWithError("Fused pointwise output differs from scalar reference.");
                return;
            }
        }
        state.counters["participants"] = static_cast<double>(
            stage == Stage::kernel || stage == Stage::pointers
                ? 1
                : cpu::executor::slots(expression.elements, compiled->unit, compiled->work)
        );
        state.counters["work"] = static_cast<double>(compiled->work);
        state.counters["unit"] = static_cast<double>(compiled->unit);
        state.counters["inputs"] = static_cast<double>(inputs.size());
        state.counters["operations"] = static_cast<double>(expression.operations);
        state.counters["plan_steps"] = static_cast<double>(plan.steps());
        state.SetItemsProcessed(state.iterations() * elements);
    }

    void executionSizes(benchmark::internal::Benchmark* benchmark) {
        for (const auto broadcast : {0, 1}) {
            benchmark->Args({1, 1, broadcast});
            benchmark->Args({32, 32, broadcast});
            benchmark->Args({2048, 64, broadcast});
            benchmark->Args({131072, 128, broadcast});
            benchmark->Args({524288, 128, broadcast});
            benchmark->Args({1048576, 128, broadcast});
        }
    }
}

BENCHMARK_CAPTURE(bmFusedExecution, pointers, Stage::pointers)
    ->Name("runtime/jit/execution/pointers")->Apply(executionSizes)->UseRealTime();
BENCHMARK_CAPTURE(bmFusedExecution, kernel, Stage::kernel)
    ->Name("runtime/jit/execution/kernel")->Apply(executionSizes)->UseRealTime();
BENCHMARK_CAPTURE(bmFusedExecution, executor, Stage::executor)
    ->Name("runtime/jit/execution/executor")->Apply(executionSizes)->UseRealTime();
BENCHMARK_CAPTURE(bmFusedExecution, direct, Stage::direct)
    ->Name("runtime/jit/execution/direct")->Apply(executionSizes)->UseRealTime();
BENCHMARK_CAPTURE(bmFusedExecution, prepared, Stage::prepared)
    ->Name("runtime/jit/execution/prepared")->Apply(executionSizes)->UseRealTime();
BENCHMARK_CAPTURE(bmFusedExecution, plan, Stage::plan)
    ->Name("runtime/jit/execution/plan")->Apply(executionSizes)->UseRealTime();
