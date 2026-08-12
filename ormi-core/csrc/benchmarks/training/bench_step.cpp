#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#if defined(ORMI_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>

#include "core/runtime/dispatch/device.h"
#include "core/runtime/fusion/group.h"
#include "core/runtime/schedule.h"

#if defined(ORMI_ENABLE_CUDA)
#include "backends/cuda/state.h"
#endif

namespace {
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    constexpr std::int64_t features = 16;
    constexpr std::int64_t width = 32;
    constexpr double learning_rate = 0.001;
    constexpr std::int64_t training_plan_steps = 23;

#if defined(ORMI_ENABLE_CUDA)
    void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + ": " + cudaGetErrorString(status)
            );
        }
    }
#endif

    struct Training {
        explicit Training(ormi::core::Device device, std::int64_t batch)
            : device(device),
              input(ormi::core::ops::full(
                  {batch, features},
                  0.25f,
                  device
              )),
              target(ormi::core::ops::full(
                  {batch, 1},
                  0.5f,
                  device
              )),
              parameters{
                  ormi::core::ops::full(
                      {features, width},
                      0.125f,
                      device
                  ),
                  ormi::core::ops::full({width}, 0.01f, device),
                  ormi::core::ops::full({width, 1}, 0.125f, device),
                  ormi::core::ops::full({1}, 0.01f, device)
              }
        {
            input->realize();
            target->realize();
            for (const auto& parameter : parameters) {
                parameter->realize();
            }
            synchronize();
        }

        Tensor loss() const {
            const auto hidden = ormi::core::ops::sin(
                ormi::core::ops::add(
                    ormi::core::ops::matmul(input, parameters[0]),
                    parameters[1]
                )
            );
            const auto prediction = ormi::core::ops::add(
                ormi::core::ops::matmul(hidden, parameters[2]),
                parameters[3]
            );
            const auto error =
                ormi::core::ops::sub(prediction, target);
            return ormi::core::ops::mean(
                ormi::core::ops::mul(error, error)
            );
        }

        std::vector<Tensor> gradients(const Tensor& loss) const {
            return ormi::core::grad(loss, parameters);
        }

        std::vector<Tensor> updates(const std::vector<Tensor>& gradients) const {
            std::vector<Tensor> result;
            result.reserve(parameters.size());
            for (std::size_t i = 0; i < parameters.size(); ++i) {
                result.push_back(ormi::core::optim::sgd(
                    parameters[i],
                    gradients[i],
                    learning_rate
                ));
            }
            return result;
        }

        void realize(const std::vector<Tensor>& tensors) const {
            for (const auto& tensor : tensors) {
                tensor->realize();
            }
        }

        void apply(const std::vector<Tensor>& updates) const {
            if (updates.size() != parameters.size()) {
                throw std::logic_error(
                    "Training update count does not match parameter count."
                );
            }

            for (std::size_t i = 0; i < parameters.size(); ++i) {
#if defined(ORMI_ENABLE_CUDA)
                if (device.name() == "cuda") {
                    check_cuda(
                        cudaMemcpyAsync(
                            parameters[i]->data(),
                            updates[i]->data(),
                            parameters[i]->bytes(),
                            cudaMemcpyDeviceToDevice,
                            ormi::backends::cuda::stream(device)
                        ),
                        "cudaMemcpyAsyncDeviceToDevice"
                    );
                    continue;
                }
#endif
                std::memcpy(
                    parameters[i]->data(),
                    updates[i]->data(),
                    parameters[i]->bytes()
                );
            }
        }

        void synchronize() const {
            ormi::core::runtime::device::synchronize(device);
        }

        ormi::core::Device device;
        Tensor input;
        Tensor target;
        std::vector<Tensor> parameters;
    };

    bool available(benchmark::State& state, bool cuda) {
        if (!cuda) {
            return true;
        }

#if defined(ORMI_ENABLE_CUDA)
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
#else
        state.SkipWithError("CUDA is not enabled.");
        return false;
#endif
    }

    ormi::core::Device device(bool cuda) {
        return cuda
            ? ormi::core::Device::from("cuda")
            : ormi::core::Device::cpu();
    }

    template <typename Run>
    void warmup_training(const Training& training, Run&& run) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
        do {
            run();
            training.synchronize();
        } while (std::chrono::steady_clock::now() < deadline);
    }

    void complete_step(const Training& training) {
        auto loss = training.loss();
        const auto gradients = training.gradients(loss);
        auto updates = training.updates(gradients);

        loss->realize();
        training.realize(updates);
        training.apply(updates);
        training.synchronize();
        benchmark::DoNotOptimize(loss.get());
        benchmark::DoNotOptimize(updates.data());
    }

    std::vector<ormi::core::Plan::Update> assignments(
        const Training& training,
        const std::vector<Tensor>& updates
    ) {
        if (updates.size() != training.parameters.size()) {
            throw std::logic_error(
                "Training update count does not match parameter count."
            );
        }

        std::vector<ormi::core::Plan::Update> result;
        result.reserve(training.parameters.size());
        for (
            std::size_t i = 0;
            i < training.parameters.size();
            ++i
        ) {
            result.push_back({
                training.parameters[i],
                updates[i]
            });
        }
        return result;
    }

    struct TrainingSchedule {
        Tensor loss;
        std::vector<Tensor> gradients;
        std::vector<Tensor> updates;
        std::vector<ormi::core::runtime::Step> steps;
    };

    TrainingSchedule schedule(
        const Training& training,
        bool group_outputs = false
    ) {
        TrainingSchedule result;
        result.loss = training.loss();
        result.gradients = training.gradients(result.loss);
        result.updates = training.updates(result.gradients);

        std::unordered_set<const ormi::core::Tensor*> planned;
        ormi::core::runtime::fusion::Groups grouped;
        if (group_outputs) {
            std::vector<ormi::core::Tensor*> roots{result.loss.get()};
            roots.reserve(1 + result.updates.size());
            for (const auto& update : result.updates) {
                roots.push_back(update.get());
            }
            grouped = ormi::core::runtime::fusion::groups(roots);
        }
        ormi::core::runtime::schedule::append(
            *result.loss,
            planned,
            result.steps,
            grouped
        );
        for (const auto& update : result.updates) {
            ormi::core::runtime::schedule::append(
                *update,
                planned,
                result.steps,
                grouped
            );
        }
        return result;
    }

    void bmTrainingConstruct(benchmark::State& state, bool cuda) {
        if (!available(state, cuda)) {
            return;
        }

        const Training training(device(cuda), state.range(0));

        for (auto _ : state) {
            auto loss = training.loss();
            auto gradients = training.gradients(loss);
            auto updates = training.updates(gradients);
            benchmark::DoNotOptimize(loss);
            benchmark::DoNotOptimize(gradients);
            benchmark::DoNotOptimize(updates);

            state.PauseTiming();
            updates.clear();
            gradients.clear();
            loss.reset();
            state.ResumeTiming();
        }
    }

    void bmTrainingGradientConstruct(benchmark::State& state, bool cuda) {
        if (!available(state, cuda)) {
            return;
        }

        const Training training(device(cuda), state.range(0));
        const auto loss = training.loss();

        for (auto _ : state) {
            auto gradients = training.gradients(loss);
            benchmark::DoNotOptimize(gradients);

            state.PauseTiming();
            gradients.clear();
            state.ResumeTiming();
        }
    }

    void bmTrainingForward(benchmark::State& state, bool cuda) {
        if (!available(state, cuda)) {
            return;
        }

        const Training training(device(cuda), state.range(0));
        training.loss()->realize();
        training.synchronize();

        for (auto _ : state) {
            state.PauseTiming();
            auto loss = training.loss();
            state.ResumeTiming();

            loss->realize();
            training.synchronize();
            benchmark::DoNotOptimize(loss);

            state.PauseTiming();
            loss.reset();
            state.ResumeTiming();
        }
    }

    void bmTrainingBackward(benchmark::State& state, bool cuda) {
        if (!available(state, cuda)) {
            return;
        }

        const Training training(device(cuda), state.range(0));
        {
            const auto loss = training.loss();
            loss->realize();
            const auto gradients = training.gradients(loss);
            training.realize(gradients);
            training.synchronize();
        }

        for (auto _ : state) {
            state.PauseTiming();
            auto loss = training.loss();
            loss->realize();
            training.synchronize();
            auto gradients = training.gradients(loss);
            state.ResumeTiming();

            training.realize(gradients);
            training.synchronize();
            benchmark::DoNotOptimize(gradients);

            state.PauseTiming();
            gradients.clear();
            loss.reset();
            state.ResumeTiming();
        }
    }

    void bmTrainingOptimizer(benchmark::State& state, bool cuda) {
        if (!available(state, cuda)) {
            return;
        }

        const Training training(device(cuda), state.range(0));
        const auto loss = training.loss();
        const auto gradients = training.gradients(loss);
        training.realize(gradients);
        training.synchronize();
        {
            const auto updates = training.updates(gradients);
            training.realize(updates);
            training.synchronize();
        }

        for (auto _ : state) {
            state.PauseTiming();
            auto updates = training.updates(gradients);
            state.ResumeTiming();

            training.realize(updates);
            training.synchronize();
            benchmark::DoNotOptimize(updates);

            state.PauseTiming();
            updates.clear();
            state.ResumeTiming();
        }
    }

    void bmTrainingApply(benchmark::State& state, bool cuda) {
        if (!available(state, cuda)) {
            return;
        }

        const Training training(device(cuda), state.range(0));
        const auto loss = training.loss();
        const auto gradients = training.gradients(loss);
        const auto updates = training.updates(gradients);
        training.realize(updates);
        training.synchronize();
        training.apply(updates);
        training.synchronize();

        for (auto _ : state) {
            training.apply(updates);
            training.synchronize();
        }
    }

    void bmTrainingStep(benchmark::State& state, bool cuda) {
        if (!available(state, cuda)) {
            return;
        }

        const Training training(device(cuda), state.range(0));
        complete_step(training);

        for (auto _ : state) {
            complete_step(training);
        }

        state.SetItemsProcessed(
            state.iterations() * state.range(0)
        );
    }

    void bmTrainingPlan(benchmark::State& state, bool cuda) {
        if (!available(state, cuda)) {
            return;
        }

        const Training training(device(cuda), state.range(0));
        complete_step(training);

        const auto loss = training.loss();
        const auto gradients = training.gradients(loss);
        const auto updates = training.updates(gradients);

        const auto prepare_start = std::chrono::steady_clock::now();
        ormi::core::Plan plan(
            {loss},
            assignments(training, updates)
        );
        const auto prepare_end = std::chrono::steady_clock::now();
        plan.run();
        training.synchronize();
        warmup_training(training, [&plan] {
            plan.run();
        });

        for (auto _ : state) {
            plan.run();
            training.synchronize();
            benchmark::DoNotOptimize(loss.get());
        }

        state.SetItemsProcessed(
            state.iterations() * state.range(0)
        );
        state.counters["prepare_us"] =
            std::chrono::duration<double, std::micro>(
                prepare_end - prepare_start
            ).count();
        state.counters["steps"] =
            static_cast<double>(plan.steps());
        state.counters["storage_buffers"] =
            static_cast<double>(plan.storage_buffers());
        state.counters["storage_bytes"] =
            static_cast<double>(plan.storage_bytes());
    }

    void bmTrainingStepSequence(
        benchmark::State& state,
        bool group_outputs
    ) {
        const Training training(
            ormi::core::Device::cpu(),
            state.range(0)
        );
        complete_step(training);

        auto scheduled = schedule(training, group_outputs);
        for (auto& step : scheduled.steps) {
            step.run();
        }
        training.apply(scheduled.updates);
        training.synchronize();

        for (auto _ : state) {
            for (auto& step : scheduled.steps) {
                step.run();
            }
            training.apply(scheduled.updates);
            training.synchronize();
            benchmark::DoNotOptimize(scheduled.loss.get());
        }

        state.SetItemsProcessed(
            state.iterations() * state.range(0)
        );
        state.counters["steps"] =
            static_cast<double>(scheduled.steps.size());
    }

    void bmTrainingPlanConstruct(benchmark::State& state) {
        const Training training(
            ormi::core::Device::cpu(),
            state.range(0)
        );
        complete_step(training);

        const auto loss = training.loss();
        const auto gradients = training.gradients(loss);
        const auto updates = training.updates(gradients);
        const auto planned_updates = assignments(training, updates);

        {
            ormi::core::Plan warmup({loss}, planned_updates);
            benchmark::DoNotOptimize(warmup.steps());
        }

        for (auto _ : state) {
            std::optional<ormi::core::Plan> plan;
            plan.emplace(
                std::vector<Tensor>{loss},
                planned_updates
            );
            benchmark::DoNotOptimize(plan->steps());

            state.PauseTiming();
            plan.reset();
            state.ResumeTiming();
        }
    }

    void bmTrainingPlanStep(
        benchmark::State& state,
        bool group_outputs
    ) {
        const Training training(
            ormi::core::Device::cpu(),
            state.range(0)
        );
        complete_step(training);

        auto scheduled = schedule(training, group_outputs);
        if (scheduled.steps.size() != training_plan_steps) {
            state.SkipWithError("The training plan step count changed.");
            return;
        }

        for (auto& step : scheduled.steps) {
            step.run();
        }

        const auto index = static_cast<std::size_t>(state.range(1));
        auto& step = scheduled.steps[index];
        auto& output = step.output();
        std::ostringstream label;
        label << output.op()->name << ' ' << output.shape();
        state.SetLabel(label.str());

        for (auto _ : state) {
            step.run();
            benchmark::DoNotOptimize(output.data());
        }

        state.counters["input_count"] =
            static_cast<double>(step.input_count());
        state.counters["output_bytes"] =
            static_cast<double>(output.bytes());
    }

    void bmTrainingGradientPlan(
        benchmark::State& state,
        std::size_t parameter
    ) {
        const Training training(
            ormi::core::Device::cpu(),
            state.range(0)
        );
        const auto loss = training.loss();
        const auto gradients = training.gradients(loss);
        if (parameter >= gradients.size()) {
            state.SkipWithError("Training gradient index is out of range.");
            return;
        }

        ormi::core::Plan plan({gradients[parameter]});
        plan.run();

        for (auto _ : state) {
            plan.run();
            benchmark::DoNotOptimize(gradients[parameter].get());
        }

        state.counters["steps"] =
            static_cast<double>(plan.steps());
        state.counters["storage_buffers"] =
            static_cast<double>(plan.storage_buffers());
        state.counters["storage_bytes"] =
            static_cast<double>(plan.storage_bytes());
    }

#if defined(ORMI_ENABLE_CUDA)
    float read_scalar(const Tensor& tensor) {
        float value = 0.0f;
        ormi::core::memory::copy(
            &value,
            ormi::core::Device::cpu(),
            tensor->data(),
            tensor->device(),
            sizeof(value)
        );
        return value;
    }

    void bmCUDATrainingGraphReplayImpl(
        benchmark::State& state,
        bool complete
    ) {
        if (!available(state, true)) {
            return;
        }

        const Training training(
            ormi::core::Device::from("cuda"),
            state.range(0)
        );

        // Capture must only see already-compiled kernels and initialized cuBLAS.
        complete_step(training);

        auto loss = training.loss();
        const auto gradients = training.gradients(loss);
        auto updates = training.updates(gradients);
        const auto prepare_start = std::chrono::steady_clock::now();
        ormi::core::Plan plan(
            {loss},
            assignments(training, updates)
        );
        const auto prepare_end = std::chrono::steady_clock::now();
        plan.run();
        training.synchronize();

        const auto capture_start = std::chrono::steady_clock::now();
        std::optional<ormi::core::Plan::Replay> replay;
        try {
            replay.emplace(plan.record());
        } catch (const std::exception& error) {
            state.SkipWithError(error.what());
            return;
        }
        const auto capture_end = std::chrono::steady_clock::now();

        replay->run();
        training.synchronize();
        const float first_loss = read_scalar(loss);
        replay->run();
        training.synchronize();
        const float second_loss = read_scalar(loss);
        if (
            !std::isfinite(first_loss) ||
            !std::isfinite(second_loss) ||
            first_loss == second_loss
        ) {
            state.SkipWithError(
                "CUDA graph replay did not advance the training state."
            );
            return;
        }

        warmup_training(training, [&replay] {
            replay->run();
        });

        for (auto _ : state) {
            try {
                replay->run();
                if (complete) {
                    training.synchronize();
                }
            } catch (const std::exception& error) {
                state.SkipWithError(error.what());
                break;
            }
            benchmark::DoNotOptimize(loss.get());
            benchmark::DoNotOptimize(updates.data());
        }

        training.synchronize();
        state.counters["prepare_us"] =
            std::chrono::duration<double, std::micro>(
                prepare_end - prepare_start
            ).count();
        state.counters["capture_us"] =
            std::chrono::duration<double, std::micro>(
                capture_end - capture_start
            ).count();
        state.counters["storage_buffers"] =
            static_cast<double>(plan.storage_buffers());
        state.counters["storage_bytes"] =
            static_cast<double>(plan.storage_bytes());
        state.SetItemsProcessed(
            state.iterations() * state.range(0)
        );
    }

    void bmCUDATrainingGraphReplay(
        benchmark::State& state,
        bool complete
    ) {
        try {
            bmCUDATrainingGraphReplayImpl(state, complete);
        } catch (const std::exception& error) {
            state.SkipWithError(error.what());
        }
    }
#endif
}

#define ORMI_TRAINING_BATCHES(BENCHMARK_NAME) \
    BENCHMARK_NAME->Arg(1)->Arg(32)->Arg(1024)->UseRealTime()

ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingConstruct, cpu, false)
        ->Name("training/cpu/step_construct")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingGradientConstruct, cpu, false)
        ->Name("training/cpu/gradient_construct")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingForward, cpu, false)
        ->Name("training/cpu/forward_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingBackward, cpu, false)
        ->Name("training/cpu/backward_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingOptimizer, cpu, false)
        ->Name("training/cpu/optimizer_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingApply, cpu, false)
        ->Name("training/cpu/parameter_apply_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingStep, cpu, false)
        ->Name("training/cpu/step_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingPlan, cpu, false)
        ->Name("training/cpu/plan_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingStepSequence, ungrouped, false)
        ->Name("training/cpu/step_sequence_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingStepSequence, grouped, true)
        ->Name("training/cpu/grouped_step_sequence_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK(bmTrainingPlanConstruct)
        ->Name("training/cpu/plan_construct")
);
BENCHMARK_CAPTURE(bmTrainingPlanStep, ungrouped, false)
    ->Name("training/cpu/plan_step")
    ->ArgsProduct({
        {1024},
        benchmark::CreateDenseRange(0, training_plan_steps - 1, 1)
    })
    ->UseRealTime();
BENCHMARK_CAPTURE(bmTrainingPlanStep, grouped, true)
    ->Name("training/cpu/grouped_plan_step")
    ->ArgsProduct({
        {1024},
        benchmark::CreateDenseRange(0, training_plan_steps - 1, 1)
    })
    ->UseRealTime();
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingGradientPlan, first_weight, 0)
        ->Name("training/cpu/gradient_plan/first_weight")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingGradientPlan, first_bias, 1)
        ->Name("training/cpu/gradient_plan/first_bias")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingGradientPlan, second_weight, 2)
        ->Name("training/cpu/gradient_plan/second_weight")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingGradientPlan, second_bias, 3)
        ->Name("training/cpu/gradient_plan/second_bias")
);

#if defined(ORMI_ENABLE_CUDA)
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingConstruct, cuda, true)
        ->Name("training/cuda/step_construct")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingGradientConstruct, cuda, true)
        ->Name("training/cuda/gradient_construct")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingForward, cuda, true)
        ->Name("training/cuda/forward_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingBackward, cuda, true)
        ->Name("training/cuda/backward_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingOptimizer, cuda, true)
        ->Name("training/cuda/optimizer_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingApply, cuda, true)
        ->Name("training/cuda/parameter_apply_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingStep, cuda, true)
        ->Name("training/cuda/step_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(bmTrainingPlan, cuda, true)
        ->Name("training/cuda/plan_complete")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(
        bmCUDATrainingGraphReplay,
        submit,
        false
    )->Name("training/cuda/graph_replay_submit")
);
ORMI_TRAINING_BATCHES(
    BENCHMARK_CAPTURE(
        bmCUDATrainingGraphReplay,
        complete,
        true
    )->Name("training/cuda/graph_replay_complete")
);
#endif

#undef ORMI_TRAINING_BATCHES
