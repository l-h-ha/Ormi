#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/op.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/executor.h"
#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/execute.h"

namespace ormi::backends::cpu::jit::execution {
    namespace {
        struct Invocation {
            Function function;
            void* const* outputs;
            const void* const* inputs;
            unsigned long long elements;
            const void* math;
            std::size_t stride;
        };

        struct Call {
            std::shared_ptr<Compiled> compiled;
            std::vector<core::Tensor*> tensors;
            std::vector<core::Tensor*> auxiliary_outputs;
            std::vector<const void*> inputs;
            std::vector<float> partials32;
            std::vector<double> partials64;
            core::DType dtype;
            std::size_t elements;
            std::size_t range;
            core::ops::ReduceOp reduction;
            std::size_t reduced;
            std::size_t partial_slots;
        };

        void invoke(
            void* opaque,
            std::size_t slot,
            std::size_t begin,
            std::size_t end
        ) {
            const auto& invocation = *static_cast<const Invocation*>(opaque);
            if (invocation.stride != 0) {
                auto* output = static_cast<void*>(
                    static_cast<std::byte*>(invocation.outputs[0]) +
                    slot * invocation.stride
                );
                void* outputs[]{output};
                invocation.function(
                    outputs,
                    invocation.inputs,
                    invocation.elements,
                    static_cast<unsigned long long>(begin),
                    static_cast<unsigned long long>(end),
                    invocation.math
                );
                return;
            }
            invocation.function(
                invocation.outputs,
                invocation.inputs,
                invocation.elements,
                static_cast<unsigned long long>(begin),
                static_cast<unsigned long long>(end),
                invocation.math
            );
        }

        template <typename T>
        T accumulate(
            const Compiled& compiled,
            std::size_t elements,
            const void* const* inputs,
            T* prepared,
            std::size_t prepared_slots
        ) {
            const auto slots = executor::slots(
                compiled.range,
                compiled.unit,
                compiled.work
            );
            if (slots == 1) {
                T result{};
                void* outputs[]{&result};
                Invocation invocation{
                    compiled.function,
                    outputs,
                    inputs,
                    static_cast<unsigned long long>(elements),
                    compiled.math,
                    0
                };
                invoke(&invocation, 0, 0, compiled.range);
                return result;
            }

            std::vector<T> local;
            auto* values = prepared;
            if (!values) {
                local.resize(slots);
                values = local.data();
            } else if (prepared_slots < slots) {
                ORMI_THROW(
                    OperationError,
                    "Prepared CPU reduction storage is smaller than its execution plan."
                );
            }
            for (std::size_t i = 0; i < slots; ++i) {
                values[i] = T{};
            }

            Invocation invocation{
                compiled.function,
                nullptr,
                inputs,
                static_cast<unsigned long long>(elements),
                compiled.math,
                sizeof(T)
            };
            void* outputs[]{values};
            invocation.outputs = outputs;
            executor::run(
                compiled.range,
                compiled.unit,
                compiled.work,
                invoke,
                &invocation
            );

            T result{};
            for (std::size_t i = 0; i < slots; ++i) {
                result += values[i];
            }
            return result;
        }

        template <typename T>
        T finalize(
            T total,
            core::ops::ReduceOp reduction,
            std::size_t reduced
        ) {
            if (reduction == core::ops::ReduceOp::mean) {
                total /= static_cast<T>(reduced);
            }
            return total;
        }

        void execute_partial(
            core::Tensor& out,
            const Compiled& compiled,
            core::DType dtype,
            std::size_t elements,
            core::ops::ReduceOp reduction,
            std::size_t reduced,
            const void* const* inputs,
            void* partials,
            std::size_t partial_slots
        ) {
            switch (dtype) {
                case core::DType::float64:
                    *static_cast<double*>(out.data()) = finalize(
                        accumulate<double>(
                            compiled,
                            elements,
                            inputs,
                            static_cast<double*>(partials),
                            partial_slots
                        ),
                        reduction,
                        reduced
                    );
                    return;
                case core::DType::float32:
                    *static_cast<float*>(out.data()) = finalize(
                        accumulate<float>(
                            compiled,
                            elements,
                            inputs,
                            static_cast<float*>(partials),
                            partial_slots
                        ),
                        reduction,
                        reduced
                    );
                    return;
                case core::DType::float16:
                    *static_cast<core::Float16*>(out.data()) = core::Float16(
                        finalize(
                            accumulate<float>(
                                compiled,
                                elements,
                                inputs,
                                static_cast<float*>(partials),
                                partial_slots
                            ),
                            reduction,
                            reduced
                        )
                    );
                    return;
                case core::DType::int32:
                case core::DType::boolean:
                    ORMI_THROW(
                        DTypeError,
                        "CPU JIT partial reduction requires a floating-point dtype."
                    );
            }

            ORMI_THROW(
                DTypeError,
                "CPU JIT partial reduction received an unknown dtype."
            );
        }

        void run_kernel(
            core::Tensor& out,
            std::span<core::Tensor* const> auxiliary_outputs,
            const Compiled& compiled,
            core::DType dtype,
            std::size_t elements,
            std::size_t range,
            core::ops::ReduceOp reduction,
            std::size_t reduced,
            const void* const* inputs,
            void* partials,
            std::size_t partial_slots
        ) {
            if (compiled.partial) {
                if (!auxiliary_outputs.empty()) {
                    ORMI_THROW(
                        OperationError,
                        "CPU JIT reductions do not support auxiliary outputs."
                    );
                }
                execute_partial(
                    out,
                    compiled,
                    dtype,
                    elements,
                    reduction,
                    reduced,
                    inputs,
                    partials,
                    partial_slots
                );
                return;
            }

            constexpr std::size_t local_capacity = 4;
            std::array<void*, local_capacity> local_outputs{};
            std::vector<void*> overflow_outputs;
            auto* outputs = local_outputs.data();
            const auto output_count = 1 + auxiliary_outputs.size();
            if (output_count > local_capacity) {
                overflow_outputs.resize(output_count);
                outputs = overflow_outputs.data();
            }
            outputs[0] = out.data();
            for (std::size_t i = 0; i < auxiliary_outputs.size(); ++i) {
                outputs[i + 1] = auxiliary_outputs[i]->data();
            }

            Invocation invocation{
                compiled.function,
                outputs,
                inputs,
                static_cast<unsigned long long>(elements),
                compiled.math,
                0
            };
            if (compiled.unit != 0) {
                executor::run(
                    range,
                    compiled.unit,
                    compiled.work,
                    invoke,
                    &invocation
                );
                return;
            }
            invoke(&invocation, 0, 0, range);
        }

        std::unique_ptr<Call> make_call(
            std::span<core::Tensor* const> auxiliary_outputs,
            const core::runtime::fusion::Expr& expression,
            std::shared_ptr<Compiled> compiled
        ) {
            const auto reduction = expression.reduction
                ? expression.reduction->op
                : core::ops::ReduceOp::sum;
            const auto reduced = expression.reduction
                ? expression.reduction->reduced.size
                : 0;
            const auto range = expression.reduction
                ? expression.reduction->output.size
                : expression.elements;
            const auto partial_slots = compiled->partial
                ? executor::slots(
                    compiled->range,
                    compiled->unit,
                    compiled->work
                )
                : 0;
            std::vector<float> partials32;
            std::vector<double> partials64;
            if (partial_slots > 1) {
                if (expression.dtype == core::DType::float64) {
                    partials64.resize(partial_slots);
                } else {
                    partials32.resize(partial_slots);
                }
            }
            return std::make_unique<Call>(Call{
                .compiled = std::move(compiled),
                .tensors = expression.inputs,
                .auxiliary_outputs = {
                    auxiliary_outputs.begin(),
                    auxiliary_outputs.end()
                },
                .inputs = std::vector<const void*>(expression.inputs.size()),
                .partials32 = std::move(partials32),
                .partials64 = std::move(partials64),
                .dtype = expression.dtype,
                .elements = expression.elements,
                .range = range,
                .reduction = reduction,
                .reduced = reduced,
                .partial_slots = partial_slots
            });
        }

        void run_call(void* state, core::Tensor& out) {
            auto& call = *static_cast<Call*>(state);
            for (std::size_t i = 0; i < call.tensors.size(); ++i) {
                call.inputs[i] = call.tensors[i]->data();
            }

            void* partials = nullptr;
            if (!call.partials32.empty()) {
                partials = call.partials32.data();
            } else if (!call.partials64.empty()) {
                partials = call.partials64.data();
            }
            run_kernel(
                out,
                call.auxiliary_outputs,
                *call.compiled,
                call.dtype,
                call.elements,
                call.range,
                call.reduction,
                call.reduced,
                call.inputs.data(),
                partials,
                call.partial_slots
            );
        }

        void drop_call(void* state) noexcept {
            delete static_cast<Call*>(state);
        }
    }

    core::Setup prepare(
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        std::shared_ptr<Compiled> compiled
    ) {
        auto call = make_call(
            auxiliary_outputs,
            expression,
            std::move(compiled)
        );
        const auto partial_bytes =
            call->partials32.size() * sizeof(float) +
            call->partials64.size() * sizeof(double);
        return core::Setup(
            call.release(),
            run_call,
            drop_call,
            partial_bytes == 0 ? 0 : 1,
            partial_bytes
        );
    }

    void run(
        core::Tensor& out,
        std::span<core::Tensor* const> auxiliary_outputs,
        const core::runtime::fusion::Expr& expression,
        const Compiled& compiled
    ) {
        constexpr std::size_t local_capacity = 8;
        std::array<const void*, local_capacity> local_inputs{};
        std::vector<const void*> overflow_inputs;
        auto* inputs = local_inputs.data();

        if (expression.inputs.size() > local_capacity) {
            overflow_inputs.resize(expression.inputs.size());
            inputs = overflow_inputs.data();
        }

        for (std::size_t i = 0; i < expression.inputs.size(); ++i) {
            inputs[i] = expression.inputs[i]->data();
        }

        const auto range = expression.reduction
            ? expression.reduction->output.size
            : expression.elements;
        run_kernel(
            out,
            auxiliary_outputs,
            compiled,
            expression.dtype,
            expression.elements,
            range,
            expression.reduction
                ? expression.reduction->op
                : core::ops::ReduceOp::sum,
            expression.reduction
                ? expression.reduction->reduced.size
                : 0,
            inputs,
            nullptr,
            0
        );
    }
}
