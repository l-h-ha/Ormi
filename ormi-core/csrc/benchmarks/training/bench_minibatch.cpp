#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include <ormi/core/autograd.h>
#include <ormi/core/ops/linalg.h>
#include <ormi/core/optim/sgd.h>
#include <ormi/core/plan.h>
#include <ormi/core/tensor.h>
#include <ormi/nn/losses/mse.h>

#include "core/runtime/dispatch/device.h"

namespace {
    using namespace ormi::core;
    using TensorPtr = std::shared_ptr<Tensor>;

    Plan training_plan(const TensorPtr& input, const TensorPtr& target, const TensorPtr& weight) {
        const auto loss = ormi::nn::mse_loss(ops::matmul(input, weight), target);
        const auto gradient = grad(loss, weight);
        return Plan({loss}, {{weight, optim::sgd(weight, gradient, 0.001)}});
    }

    void bmMinibatch(benchmark::State& state, const char* backend, bool rebuild) {
        const auto device = Device::from(backend);
        const auto batch = state.range(0);
        const Shape input_shape{batch, 16};
        const Shape target_shape{batch, 4};
        std::array<std::vector<float>, 2> values{
            std::vector<float>(input_shape.numel(), 0.25f),
            std::vector<float>(input_shape.numel(), -0.2f)
        };
        std::array<std::vector<float>, 2> labels{
            std::vector<float>(target_shape.numel(), 0.5f),
            std::vector<float>(target_shape.numel(), 0.1f)
        };
        auto input = std::make_shared<Tensor>(device, input_shape, DType::float32);
        auto target = std::make_shared<Tensor>(device, target_shape, DType::float32);
        const auto weight = std::make_shared<Tensor>(device, Shape{16, 4}, DType::float32);
        const std::vector<float> weights(weight->numel(), 0.05f);
        weight->copy_from(std::as_bytes(std::span(weights)));
        input->copy_from(std::as_bytes(std::span(values[0])));
        target->copy_from(std::as_bytes(std::span(labels[0])));
        auto plan = training_plan(input, target, weight);
        plan.run();
        runtime::device::synchronize(device);

        std::size_t index = 0;
        for (auto _ : state) {
            index ^= 1;
            if (rebuild) {
                input = std::make_shared<Tensor>(device, input_shape, DType::float32);
                target = std::make_shared<Tensor>(device, target_shape, DType::float32);
            }
            input->copy_from(std::as_bytes(std::span(values[index])));
            target->copy_from(std::as_bytes(std::span(labels[index])));
            if (rebuild) {
                plan = training_plan(input, target, weight);
            }
            plan.run();
            runtime::device::synchronize(device);
        }
        state.SetItemsProcessed(state.iterations() * batch);
    }
}

BENCHMARK_CAPTURE(bmMinibatch, cpu_reuse, "cpu", false)
    ->Name("training/minibatch_cpu_reuse")
    ->Arg(1)
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();
BENCHMARK_CAPTURE(bmMinibatch, cpu_rebuild, "cpu", true)
    ->Name("training/minibatch_cpu_rebuild")
    ->Arg(1)
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();

#ifdef ORMI_ENABLE_CUDA
BENCHMARK_CAPTURE(bmMinibatch, cuda_reuse, "cuda", false)
    ->Name("training/minibatch_cuda_reuse")
    ->Arg(1)
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();
BENCHMARK_CAPTURE(bmMinibatch, cuda_rebuild, "cuda", true)
    ->Name("training/minibatch_cuda_rebuild")
    ->Arg(1)
    ->Arg(32)
    ->Arg(1024)
    ->UseRealTime();
#endif
