#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/setup.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/executor.h"
#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/execute.h"

namespace ormi::backends::cpu::jit::execution {
    namespace {
        constexpr std::size_t input_capacity = 8;
        constexpr std::size_t output_capacity = 4;

        struct Kernel {
            Kernel(const core::runtime::fusion::Expr& expression, const Compiled& compiled)
                : compiled(compiled),
                  dtype(expression.dtype),
                  elements(expression.elements),
                  range(expression.elements),
                  reduction(core::ops::ReduceOp::sum),
                  reduced(0) {
                if (expression.reduction) {
                    range = expression.reduction->output.size;
                    reduction = expression.reduction->op;
                    reduced = expression.reduction->reduced.size;
                }
                if (compiled.partial) {
                    range = compiled.range;
                }
            }

            const Compiled& compiled;
            core::DType dtype;
            std::size_t elements;
            // Partial kernels partition the reduction. Other kernels partition outputs.
            std::size_t range;
            core::ops::ReduceOp reduction;
            std::size_t reduced;
        };

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
            Kernel kernel;
            std::vector<core::Tensor*> tensors;
            std::vector<core::Tensor*> auxiliary_outputs;
            std::vector<const void*> inputs;
            std::vector<void*> overflow_outputs;
            std::vector<float> partials32;
            std::vector<double> partials64;
        };

        void invoke(void* opaque, std::size_t slot, std::size_t begin, std::size_t end) {
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
        T accumulate(const Kernel& kernel, const void* const* inputs, std::span<T> prepared) {
            const auto& compiled = kernel.compiled;
            const auto slots = executor::slots(kernel.range, compiled.unit, compiled.work);
            if (slots == 1) {
                T result{};
                void* outputs[]{&result};
                Invocation invocation{
                    compiled.function,
                    outputs,
                    inputs,
                    static_cast<unsigned long long>(kernel.elements),
                    compiled.math,
                    0
                };
                invoke(&invocation, 0, 0, kernel.range);
                return result;
            }

            std::vector<T> local;
            auto* values = prepared.data();
            if (prepared.empty()) {
                local.resize(slots);
                values = local.data();
            } else if (prepared.size() < slots) {
                ORMI_THROW(
                    OperationError,
                    "Prepared CPU reduction storage is smaller than its execution plan."
                );
            }
            for (std::size_t i = 0; i < slots; ++i) {
                values[i] = T{};
            }

            void* outputs[]{values};
            Invocation invocation{
                compiled.function,
                outputs,
                inputs,
                static_cast<unsigned long long>(kernel.elements),
                compiled.math,
                sizeof(T)
            };
            executor::run(
                kernel.range,
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
        T finalize(T total, const Kernel& kernel) {
            if (kernel.reduction == core::ops::ReduceOp::mean) {
                total /= static_cast<T>(kernel.reduced);
            }
            return total;
        }

        void execute_partial(
            void* out,
            const Kernel& kernel,
            const void* const* inputs,
            std::span<float> partials32,
            std::span<double> partials64
        ) {
            switch (kernel.dtype) {
                case core::DType::float64:
                    *static_cast<double*>(out) = finalize(
                        accumulate<double>(kernel, inputs, partials64), kernel
                    );
                    return;
                case core::DType::float32:
                    *static_cast<float*>(out) = finalize(
                        accumulate<float>(kernel, inputs, partials32), kernel
                    );
                    return;
                case core::DType::float16:
                    *static_cast<core::Float16*>(out) = core::Float16(
                        finalize(
                            accumulate<float>(kernel, inputs, partials32), kernel
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
            const Kernel& kernel,
            std::span<void* const> outputs,
            const void* const* inputs,
            std::span<float> partials32 = {},
            std::span<double> partials64 = {}
        ) {
            const auto& compiled = kernel.compiled;
            if (compiled.partial) {
                if (outputs.size() != 1) {
                    ORMI_THROW(
                        OperationError,
                        "CPU JIT reductions do not support auxiliary outputs."
                    );
                }
                execute_partial(
                    outputs[0], kernel, inputs, partials32, partials64
                );
                return;
            }

            Invocation invocation{
                compiled.function,
                outputs.data(),
                inputs,
                static_cast<unsigned long long>(kernel.elements),
                compiled.math,
                0
            };
            if (compiled.unit != 0) {
                executor::run(
                    kernel.range,
                    compiled.unit,
                    compiled.work,
                    invoke,
                    &invocation
                );
                return;
            }
            invoke(&invocation, 0, 0, kernel.range);
        }

        void run_call(void* state, core::Tensor& out) {
            auto& call = *static_cast<Call*>(state);
            std::array<void*, output_capacity> local_outputs{};
            auto* outputs = call.overflow_outputs.empty()
                ? local_outputs.data() : call.overflow_outputs.data();
            for (std::size_t i = 0; i < call.tensors.size(); ++i) {
                call.inputs[i] = call.tensors[i]->data();
            }
            outputs[0] = out.data();
            for (std::size_t i = 0; i < call.auxiliary_outputs.size(); ++i) {
                outputs[i + 1] = call.auxiliary_outputs[i]->data();
            }
            run_kernel(
                call.kernel,
                {outputs, 1 + call.auxiliary_outputs.size()},
                call.inputs.data(),
                call.partials32,
                call.partials64
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
        const Kernel kernel(expression, *compiled);
        std::vector<float> partials32;
        std::vector<double> partials64;
        if (kernel.compiled.partial) {
            const auto slots = executor::slots(
                kernel.range, kernel.compiled.unit, kernel.compiled.work
            );
            if (slots > 1) {
                if (kernel.dtype == core::DType::float64) {
                    partials64.resize(slots);
                } else {
                    partials32.resize(slots);
                }
            }
        }
        const auto partial_bytes =
            partials32.size() * sizeof(float) + partials64.size() * sizeof(double);
        auto call = std::make_unique<Call>(Call{
            .compiled = std::move(compiled),
            .kernel = kernel,
            .tensors = expression.inputs,
            .auxiliary_outputs = {auxiliary_outputs.begin(), auxiliary_outputs.end()},
            .inputs = std::vector<const void*>(expression.inputs.size()),
            .overflow_outputs = {},
            .partials32 = std::move(partials32),
            .partials64 = std::move(partials64)
        });
        const auto output_count = 1 + auxiliary_outputs.size();
        if (output_count > output_capacity) {
            call->overflow_outputs.resize(output_count);
        }
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
        const Kernel kernel(expression, compiled);
        std::array<const void*, input_capacity> local_inputs{};
        std::array<void*, output_capacity> local_outputs{};
        std::vector<const void*> overflow_inputs;
        std::vector<void*> overflow_outputs;
        auto* inputs = local_inputs.data();
        auto* outputs = local_outputs.data();
        const auto output_count = 1 + auxiliary_outputs.size();
        if (expression.inputs.size() > input_capacity) {
            overflow_inputs.resize(expression.inputs.size());
            inputs = overflow_inputs.data();
        }
        if (output_count > output_capacity) {
            overflow_outputs.resize(output_count);
            outputs = overflow_outputs.data();
        }
        for (std::size_t i = 0; i < expression.inputs.size(); ++i) {
            inputs[i] = expression.inputs[i]->data();
        }

        outputs[0] = out.data();
        for (std::size_t i = 0; i < auxiliary_outputs.size(); ++i) {
            outputs[i + 1] = auxiliary_outputs[i]->data();
        }
        run_kernel(kernel, {outputs, output_count}, inputs);
    }
}
