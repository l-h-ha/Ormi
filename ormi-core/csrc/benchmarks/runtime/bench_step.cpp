#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "core/runtime/step.h"

namespace {
    namespace core = ormi::core;
    using Tensor = std::shared_ptr<core::Tensor>;
    using Step = core::runtime::Step;
    using Expr = core::runtime::fusion::Expr;

    void write_outputs(core::Tensor& output, std::span<core::Tensor* const> auxiliary) {
        *static_cast<float*>(output.data()) = 1.0f;
        for (std::size_t index = 0; index < auxiliary.size(); ++index) {
            *static_cast<float*>(auxiliary[index]->data()) = static_cast<float>(index + 2);
        }
    }

    void execute(core::Tensor& output, std::span<core::Tensor* const> auxiliary, const Expr&) {
        write_outputs(output, auxiliary);
    }

    core::Setup prepare(core::Tensor&, std::span<core::Tensor* const> auxiliary, const Expr&) {
        return core::Setup(
            new std::vector<core::Tensor*>{auxiliary.begin(), auxiliary.end()},
            [](void* state, core::Tensor& output) {
                write_outputs(output, *static_cast<std::vector<core::Tensor*>*>(state));
            },
            [](void* state) noexcept {
                delete static_cast<std::vector<core::Tensor*>*>(state);
            }
        );
    }

    std::vector<Tensor> outputs(const Tensor& input, std::size_t count) {
        std::vector<Tensor> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            result.push_back(core::ops::neg(input));
        }
        return result;
    }

    Step step(const std::vector<Tensor>& outputs, bool prepared) {
        Expr expression{};
        expression.dtype = core::DType::float32;
        expression.elements = 1;
        std::vector<core::Tensor*> auxiliary;
        auxiliary.reserve(outputs.size() - 1);
        for (std::size_t index = 1; index < outputs.size(); ++index) {
            auxiliary.push_back(outputs[index].get());
        }
        if (prepared) {
            return Step::deferred_fusion(
                *outputs.front(), prepare, nullptr, execute,
                std::move(expression), std::move(auxiliary)
            );
        }
        return Step::fusion(
            *outputs.front(), execute, std::move(expression), std::move(auxiliary)
        );
    }

    bool valid(const std::vector<Tensor>& outputs) {
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            if (!outputs[index]->realized()) {
                return false;
            }
            if (*static_cast<const float*>(outputs[index]->data()) != static_cast<float>(index + 1)) {
                return false;
            }
        }
        return true;
    }

    void bmStepReady(benchmark::State& state) {
        const auto input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{}, core::DType::float32
        );
        const auto values = outputs(input, static_cast<std::size_t>(state.range(0)));
        auto current = step(values, state.range(1) != 0);
        current.run();

        for (auto _ : state) {
            current.run();
            benchmark::ClobberMemory();
        }
        if (!valid(values)) {
            state.SkipWithError("Step returned incorrect outputs.");
        }
    }

    void first_run(benchmark::State& state, bool partial) {
        const auto input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{}, core::DType::float32
        );
        for (auto _ : state) {
            bool correct = false;
            state.PauseTiming();
            {
                const auto values = outputs(input, static_cast<std::size_t>(state.range(0)));
                if (partial) {
                    for (std::size_t index = 0; index < values.size(); index += 2) {
                        auto initialize = Step::compute(*values[index], [](core::Tensor& output) {
                            *static_cast<float*>(output.data()) = 0.0f;
                        });
                        initialize.run();
                    }
                }
                auto current = step(values, state.range(1) != 0);
                current.prepare();
                state.ResumeTiming();

                current.run();
                benchmark::ClobberMemory();

                state.PauseTiming();
                correct = valid(values);
            }
            state.ResumeTiming();
            if (!correct) {
                state.SkipWithError("Step returned incorrect outputs.");
                break;
            }
        }
    }

    void bmStepFirstRun(benchmark::State& state) {
        first_run(state, false);
    }

    void bmStepPartialOutputs(benchmark::State& state) {
        first_run(state, true);
    }
}

BENCHMARK(bmStepReady)
    ->Name("runtime/step/ready")
    ->ArgsProduct({{1, 2, 4, 8}, {0, 1}})
    ->ArgNames({"outputs", "prepared"})
    ->UseRealTime();

BENCHMARK(bmStepFirstRun)
    ->Name("runtime/step/first_run")
    ->ArgsProduct({{1, 2, 4, 8}, {0, 1}})
    ->ArgNames({"outputs", "prepared"})
    ->UseRealTime();

BENCHMARK(bmStepPartialOutputs)
    ->Name("runtime/step/partial_outputs")
    ->ArgsProduct({{2, 4, 8}, {0, 1}})
    ->ArgNames({"outputs", "prepared"})
    ->UseRealTime();
