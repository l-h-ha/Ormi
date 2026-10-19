#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <ormi/core/error.h>
#include <ormi/core/setup.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/allocator.h"
#include "backends/cuda/check.cuh"
#include "backends/cuda/fusion/execute.h"
#include "backends/cuda/fusion/module.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda::fusion::execution {
    namespace {
        std::size_t accumulator_bytes(core::DType dtype) {
            return dtype == core::DType::float64
                ? sizeof(double)
                : sizeof(float);
        }

        int finish_threads(std::size_t parts) {
            int threads = 1;
            while (threads < 256 && static_cast<std::size_t>(threads) < parts) {
                threads *= 2;
            }
            return threads;
        }

        struct Call {
            std::shared_ptr<Compiled> compiled;
            State* cuda_state;
            core::memory::Allocator* memory;
            std::vector<core::Tensor*> tensors;
            std::vector<core::Tensor*> auxiliary_outputs;
            std::vector<const void*> inputs;
            std::vector<void*> outputs;
            std::vector<void*> arguments;
            unsigned long long parts;
            unsigned long long count;
            int blocks;
            int threads;
            int finish_blocks;
            int finish_threads;
            unsigned int shared_bytes;
            unsigned int finish_shared_bytes;
            bool reduction;
            bool direct;
            void* partials;
            std::size_t partials_bytes;

            ~Call() noexcept {
                if (memory) {
                    memory->deallocate(partials, partials_bytes);
                }
            }
        };

        void drop(void* state) noexcept {
            delete static_cast<Call*>(state);
        }

        std::unique_ptr<Call> make_call(
            core::Tensor& out,
            std::span<core::Tensor* const> auxiliary_outputs,
            const core::runtime::fusion::Expr& expression,
            State& cuda_state,
            std::shared_ptr<Compiled> compiled
        ) {
            // Resolve launch settings and workspace requirements before creating the call.
            const auto reduction = expression.reduction.has_value();
            const auto elements = static_cast<unsigned long long>(
                expression.elements
            );
            auto parts = 1ull;
            auto tasks = elements;
            auto threads = compiled->threads;

            if (compiled->tiled) {
                tasks = compiled->tiled->tasks;
            } else if (reduction) {
                const auto outputs = static_cast<unsigned long long>(
                    expression.reduction->output.size
                );
                const auto reduced = static_cast<unsigned long long>(
                    expression.reduction->reduced.size
                );
                threads = std::min(compiled->threads, finish_threads(reduced));

                const auto values_per_part =
                    static_cast<unsigned long long>(threads) * 8;
                const auto required_parts =
                    (reduced + values_per_part - 1) / values_per_part;
                const auto available_parts = outputs >= 65535ull
                    ? 1ull
                    : std::max(1ull, 65535ull / outputs);
                parts = std::min(required_parts, available_parts);
                tasks = outputs * parts;
            }

            const auto blocks = static_cast<int>(
                std::min<unsigned long long>(
                    reduction
                        ? tasks
                        : (
                            elements +
                            static_cast<unsigned long long>(threads) - 1
                        ) / static_cast<unsigned long long>(threads),
                    reduction
                        ? 65535ull
                        : static_cast<unsigned long long>(compiled->max_blocks)
                )
            );
            const auto value_bytes = accumulator_bytes(expression.dtype);
            auto direct = !reduction;
            if (reduction) {
                direct = compiled->tiled.has_value() || (
                    expression.reduction->op == core::ops::ReduceOp::sum &&
                    parts == 1 &&
                    expression.dtype != core::DType::float16
                );
            }
            if (
                reduction &&
                !direct &&
                tasks > std::numeric_limits<std::size_t>::max() / value_bytes
            ) {
                ORMI_THROW(
                    MemoryError,
                    "CUDA fusion workspace size overflow."
                );
            }
            const auto partials_bytes = reduction && !direct
                ? static_cast<std::size_t>(tasks) * value_bytes
                : std::size_t{0};
            auto* memory = partials_bytes != 0
                ? allocator(out.device())
                : nullptr;
            const auto outputs = reduction
                ? static_cast<unsigned long long>(
                    expression.reduction->output.size
                )
                : 0ull;
            const auto shared_bytes = reduction && !compiled->tiled
                ? static_cast<unsigned int>(
                    static_cast<std::size_t>(threads) * value_bytes
                )
                : 0u;
            const auto finish_block_count = reduction && !direct
                ? static_cast<int>(std::min(outputs, 65535ull))
                : 0;
            const auto finish_thread_count = reduction && !direct
                ? finish_threads(parts)
                : 0;
            const auto finish_shared_bytes = static_cast<unsigned int>(
                static_cast<std::size_t>(finish_thread_count) * value_bytes
            );

            // Own all setup resources before allocation or argument binding can fail.
            auto result = std::unique_ptr<Call>(new Call{
                .compiled = std::move(compiled),
                .cuda_state = &cuda_state,
                .memory = memory,
                .tensors = expression.inputs,
                .auxiliary_outputs = {
                    auxiliary_outputs.begin(),
                    auxiliary_outputs.end()
                },
                .inputs = std::vector<const void*>(expression.inputs.size()),
                .outputs = std::vector<void*>(1 + auxiliary_outputs.size()),
                .parts = parts,
                .count = reduction ? parts : elements,
                .blocks = blocks,
                .threads = threads,
                .finish_blocks = finish_block_count,
                .finish_threads = finish_thread_count,
                .shared_bytes = shared_bytes,
                .finish_shared_bytes = finish_shared_bytes,
                .reduction = reduction,
                .direct = direct,
                .partials = nullptr,
                .partials_bytes = partials_bytes
            });
            if (result->memory) {
                result->partials = result->memory->allocate(partials_bytes);
            }
            // These addresses stay valid: the heap-owned call and pointer vectors no longer move.
            result->arguments.reserve(
                result->outputs.size() + result->inputs.size() + 1
            );
            for (auto& output : result->outputs) {
                result->arguments.push_back(&output);
            }
            for (auto& input : result->inputs) {
                result->arguments.push_back(&input);
            }
            result->arguments.push_back(&result->count);
            return result;
        }

        void launch_kernel(Call& call, core::Tensor& out) {
            for (std::size_t i = 0; i < call.tensors.size(); ++i) {
                call.inputs[i] = call.tensors[i]->data();
            }
            call.outputs[0] = call.direct ? out.data() : call.partials;
            for (std::size_t i = 0; i < call.auxiliary_outputs.size(); ++i) {
                call.outputs[i + 1] = call.auxiliary_outputs[i]->data();
            }

            check_driver(
                cuLaunchKernel(
                    call.compiled->function,
                    call.blocks,
                    1,
                    1,
                    call.threads,
                    1,
                    1,
                    call.shared_bytes,
                    reinterpret_cast<CUstream>(call.cuda_state->stream()),
                    call.arguments.data(),
                    nullptr
                ),
                "cuLaunchKernel"
            );

            if (call.reduction && !call.direct) {
                void* output = out.data();
                void* finish_arguments[]{
                    &output,
                    &call.partials,
                    &call.parts
                };
                check_driver(
                    cuLaunchKernel(
                        call.compiled->finish,
                        call.finish_blocks,
                        1,
                        1,
                        call.finish_threads,
                        1,
                        1,
                        call.finish_shared_bytes,
                        reinterpret_cast<CUstream>(call.cuda_state->stream()),
                        finish_arguments,
                        nullptr
                    ),
                    "cuLaunchKernel"
                );
            }
        }

        void launch(void* state, core::Tensor& out) {
            auto& call = *static_cast<Call*>(state);
            check(
                cudaSetDevice(call.cuda_state->index()),
                "cudaSetDeviceFusion"
            );
            launch_kernel(call, out);
        }
    }

    core::Setup prepare(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        State& cuda_state,
        std::shared_ptr<Compiled> compiled
    ) {
        auto call = make_call(
            out,
            auxiliary_outputs,
            expression,
            cuda_state,
            std::move(compiled)
        );
        const auto buffers = call->partials ? std::size_t{1} : std::size_t{0};
        const auto bytes = call->partials_bytes;
        return core::Setup(call.release(), launch, drop, buffers, bytes);
    }

    void run(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        State& cuda_state,
        std::shared_ptr<Compiled> compiled
    ) {
        auto call = make_call(
            out,
            auxiliary_outputs,
            expression,
            cuda_state,
            std::move(compiled)
        );
        launch_kernel(*call, out);
    }
}
