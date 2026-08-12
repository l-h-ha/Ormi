#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/device.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/group.h"
#include "core/runtime/schedule.h"
#include "core/runtime/step.h"

namespace {
    using Clock = std::chrono::steady_clock;
    using Tensor = std::shared_ptr<ormi::core::Tensor>;

    struct Dense {
        Dense(
            std::int64_t batch,
            std::int64_t features,
            std::int64_t width
        ) : input(ormi::core::ops::full(
                {batch, features},
                0.25f,
                ormi::core::Device::cpu()
            )),
            target(ormi::core::ops::full(
                {batch, width},
                0.5f,
                ormi::core::Device::cpu()
            )),
            weight(ormi::core::ops::full(
                {features, width},
                0.125f,
                ormi::core::Device::cpu()
            )),
            bias(ormi::core::ops::full(
                {width},
                0.01f,
                ormi::core::Device::cpu()
            )) {
            input->realize();
            target->realize();
            weight->realize();
            bias->realize();
        }

        Tensor input;
        Tensor target;
        Tensor weight;
        Tensor bias;
    };

    struct DenseSchedule {
        Tensor loss;
        std::vector<Tensor> gradients;
        std::vector<ormi::core::runtime::Step> steps;
    };

    DenseSchedule schedule(const Dense& dense) {
        DenseSchedule result;
        const auto error = ormi::core::ops::sub(
            ormi::core::ops::add(
                ormi::core::ops::matmul(dense.input, dense.weight),
                dense.bias
            ),
            dense.target
        );
        result.loss = ormi::core::ops::mean(
            ormi::core::ops::mul(error, error)
        );
        result.gradients = ormi::core::grad(
            result.loss,
            {dense.weight, dense.bias}
        );

        std::vector<ormi::core::Tensor*> roots{result.loss.get()};
        roots.reserve(1 + result.gradients.size());
        for (const auto& gradient : result.gradients) {
            roots.push_back(gradient.get());
        }

        const auto groups = ormi::core::runtime::fusion::groups(roots);
        std::unordered_set<const ormi::core::Tensor*> completed;
        for (auto* root : roots) {
            ormi::core::runtime::schedule::append(
                *root,
                completed,
                result.steps,
                groups
            );
        }
        ormi::core::runtime::prepare(result.steps);
        return result;
    }

    double milliseconds(Clock::duration duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    double median(std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        const auto middle = samples.size() / 2;
        if (samples.size() % 2 != 0) {
            return samples[middle];
        }
        return (samples[middle - 1] + samples[middle]) * 0.5;
    }

    std::string step_name(
        std::size_t index,
        const ormi::core::runtime::Step& step
    ) {
        std::ostringstream name;
        name << "step_";
        if (index < 10) {
            name << '0';
        }
        name << index << '_' << step.output().op()->name << "_ms";
        return name.str();
    }

    std::string step_labels(
        const std::vector<ormi::core::runtime::Step>& steps
    ) {
        std::ostringstream labels;
        for (std::size_t index = 0; index < steps.size(); ++index) {
            if (index != 0) {
                labels << "; ";
            }
            const auto& step = steps[index];
            labels << index << '=' << step.output().op()->name
                << step.output().shape();
            if (step.output_count() != 1) {
                labels << " outputs=" << step.output_count();
            }
        }
        return labels.str();
    }

    void bmDensePlanProfile(benchmark::State& state) {
        const Dense dense(state.range(0), state.range(1), state.range(2));
        auto scheduled = schedule(dense);

        for (auto& step : scheduled.steps) {
            step.run();
        }

        std::vector<std::vector<double>> samples(scheduled.steps.size());
        std::vector<double> iteration(scheduled.steps.size());
        std::vector<double> totals;
        std::vector<double> gaps;
        for (auto _ : state) {
            const auto total_start = Clock::now();
            for (std::size_t index = 0; index < scheduled.steps.size(); ++index) {
                const auto start = Clock::now();
                scheduled.steps[index].run();
                const auto end = Clock::now();
                iteration[index] = milliseconds(end - start);
            }
            const auto total = milliseconds(Clock::now() - total_start);

            double accounted = 0.0;
            for (std::size_t index = 0; index < samples.size(); ++index) {
                samples[index].push_back(iteration[index]);
                accounted += iteration[index];
            }
            totals.push_back(total);
            gaps.push_back(total - accounted);
            benchmark::DoNotOptimize(scheduled.loss->data());
        }

        for (std::size_t index = 0; index < scheduled.steps.size(); ++index) {
            const auto value = median(std::move(samples[index]));
            state.counters[step_name(index, scheduled.steps[index])] = value;
        }
        const auto total = median(std::move(totals));
        state.counters["profile_total_ms"] = total;
        state.counters["profile_gap_ms"] = median(std::move(gaps));
        state.counters["steps"] = static_cast<double>(scheduled.steps.size());
        state.SetLabel(step_labels(scheduled.steps));
    }
}

BENCHMARK(bmDensePlanProfile)
    ->Name("training/cpu/dense_plan/profile")
    ->Args({2048, 512, 512})
    ->UseRealTime();
