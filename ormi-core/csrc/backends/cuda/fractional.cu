#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <cuda_runtime_api.h>

#include <ormi/core/attrs.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/allocator.h"
#include "backends/cuda/check.cuh"
#include "backends/cuda/elementwise/index.cuh"
#include "backends/cuda/fractional.h"
#include "backends/cuda/fractional/l2.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        struct Prepared {
            core::ops::FractionalOp op;
            core::memory::Allocator* memory;
            cudaStream_t stream;
            double* weights = nullptr;
            std::size_t weights_bytes = 0;
            void* workspace = nullptr;
            std::size_t workspace_bytes = 0;
        };

        std::size_t initial_axis_stride(
            const core::Tensor& initial_derivative,
            std::size_t axis
        ) {
            std::size_t result = 1;
            for (std::size_t i = axis; i < initial_derivative.rank(); ++i) {
                result *= static_cast<std::size_t>(
                    initial_derivative.shape()[i]
                );
            }
            return result;
        }

        template <typename T>
        __device__ T l1_adjoint(
            const T* input,
            const double* weights,
            std::size_t base,
            std::size_t coordinate,
            std::size_t count,
            std::size_t input_axis_stride
        ) {
            if (coordinate == 0) {
                if (count == 1) {
                    return T{};
                }

                T value{};
                auto weight = static_cast<T>(weights[0]);
                for (std::size_t n = 1; n < count; ++n) {
                    value -= weight * input[base + n * input_axis_stride];
                    if (n < count - 1) {
                        weight += static_cast<T>(weights[n]);
                    }
                }
                return value;
            }

            auto value = static_cast<T>(weights[0]) *
                input[base + coordinate * input_axis_stride];
            for (std::size_t n = coordinate + 1; n < count; ++n) {
                value += static_cast<T>(weights[n - coordinate]) *
                    input[base + n * input_axis_stride];
            }
            return value;
        }

        template <typename T>
        __device__ T l2_adjoint(
            const T* input,
            const double* weights,
            std::size_t base,
            std::size_t coordinate,
            std::size_t count,
            std::size_t input_axis_stride
        ) {
            if (coordinate < 2) {
                T value{};
                T previous{};
                T previous_previous{};

                for (std::size_t n = 1; n < count; ++n) {
                    T current{};
                    if (n == 1) {
                        current = static_cast<T>(weights[0]);
                    } else {
                        current =
                            static_cast<T>(weights[n - 1]) +
                            static_cast<T>(2) * previous -
                            previous_previous;
                    }

                    const auto coefficient = coordinate == 0
                        ? -static_cast<T>(2) * current + previous
                        : static_cast<T>(2) * current -
                            static_cast<T>(2) * previous +
                            previous_previous;
                    value += coefficient *
                        input[base + n * input_axis_stride];
                    previous_previous = previous;
                    previous = current;
                }
                return value;
            }

            auto value = static_cast<T>(weights[0]) *
                input[base + coordinate * input_axis_stride];
            for (std::size_t n = coordinate + 1; n < count; ++n) {
                value += static_cast<T>(weights[n - coordinate]) *
                    input[base + n * input_axis_stride];
            }
            return value;
        }

        template <typename T, core::ops::FractionalOp Op>
        __global__ void fractional_kernel(
            T* out,
            const T* input,
            const T* initial_derivative,
            const double* weights,
            std::size_t elements,
            std::size_t count,
            std::size_t axis_stride,
            std::size_t input_axis_stride,
            double spacing,
            bool second_order,
            elementwise::Index input_index,
            elementwise::Index initial_index
        ) {
            std::size_t linear =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                if constexpr (
                    Op == core::ops::FractionalOp::caputo_initial_adjoint
                ) {
                    const auto outer = linear / axis_stride;
                    const auto inner = linear % axis_stride;
                    const auto logical_base =
                        outer * count * axis_stride + inner;
                    T value{};

                    for (std::size_t n = 1; n < count; ++n) {
                        value -= static_cast<T>(2.0 * spacing * weights[n - 1]) *
                            input[elementwise::storage_offset(
                                logical_base + n * axis_stride,
                                input_index
                            )];
                    }
                    out[linear] = value;
                    continue;
                }

                const auto coordinate = linear / axis_stride % count;

                if constexpr (Op == core::ops::FractionalOp::caputo_initial) {
                    if (coordinate == 0) {
                        out[linear] = T{};
                        continue;
                    }

                    const auto outer = linear / (count * axis_stride);
                    const auto inner = linear % axis_stride;
                    const auto initial_offset = elementwise::storage_offset(
                        outer * axis_stride + inner,
                        input_index
                    );
                    out[linear] = static_cast<T>(
                        -2.0 * spacing * weights[coordinate - 1]
                    ) * input[initial_offset];
                    continue;
                }

                const auto input_offset =
                    elementwise::storage_offset(linear, input_index);
                const auto base =
                    input_offset - coordinate * input_axis_stride;

                if constexpr (Op == core::ops::FractionalOp::caputo) {
                    T value{};

                    if (!second_order) {
                        for (std::size_t k = 0; k < coordinate; ++k) {
                            const auto upper =
                                base + (coordinate - k) * input_axis_stride;
                            const auto lower = upper - input_axis_stride;
                            value += static_cast<T>(weights[k]) *
                                (input[upper] - input[lower]);
                        }
                    } else {
                        for (std::size_t k = 0; k < coordinate; ++k) {
                            const auto difference = coordinate - 1 - k;
                            T second_difference{};

                            if (difference == 0) {
                                second_difference = static_cast<T>(2) * (
                                    input[base + input_axis_stride] - input[base]
                                );
                                if (initial_derivative) {
                                    const auto outer =
                                        linear / (count * axis_stride);
                                    const auto inner = linear % axis_stride;
                                    const auto initial_offset =
                                        elementwise::storage_offset(
                                            outer * axis_stride + inner,
                                            initial_index
                                        );
                                    second_difference -=
                                        static_cast<T>(2.0 * spacing) *
                                        initial_derivative[initial_offset];
                                }
                            } else {
                                const auto center =
                                    base + difference * input_axis_stride;
                                second_difference =
                                    input[center + input_axis_stride] -
                                    static_cast<T>(2) * input[center] +
                                    input[center - input_axis_stride];
                            }
                            value += static_cast<T>(weights[k]) *
                                second_difference;
                        }
                    }
                    out[linear] = value;
                } else if (second_order) {
                    out[linear] = l2_adjoint(
                        input,
                        weights,
                        base,
                        coordinate,
                        count,
                        input_axis_stride
                    );
                } else {
                    out[linear] = l1_adjoint(
                        input,
                        weights,
                        base,
                        coordinate,
                        count,
                        input_axis_stride
                    );
                }
            }
        }

        template <typename T, core::ops::FractionalOp Op>
        void launch(
            core::Tensor& out,
            const core::Tensor& input,
            const core::Tensor* initial_derivative,
            const core::Caputo& attrs,
            const double* weights,
            cudaStream_t stream
        ) {
            const auto axis = static_cast<std::size_t>(attrs.axis);
            const auto initial_output =
                Op == core::ops::FractionalOp::caputo_initial_adjoint;
            const auto axis_stride = initial_output
                ? initial_axis_stride(out, axis)
                : static_cast<std::size_t>(out.strides()[axis]);
            const auto input_axis_stride =
                Op == core::ops::FractionalOp::caputo_initial || initial_output
                    ? 0
                    : static_cast<std::size_t>(input.strides()[axis]);

            fractional_kernel<T, Op><<<
                elementwise::blocks(out.numel()),
                elementwise::threads,
                0,
                stream
            >>>(
                static_cast<T*>(out.data()),
                static_cast<const T*>(input.data()),
                initial_derivative
                    ? static_cast<const T*>(initial_derivative->data())
                    : nullptr,
                weights,
                out.numel(),
                attrs.count,
                axis_stride,
                input_axis_stride,
                attrs.spacing,
                attrs.alpha > 1.0,
                elementwise::make_index(input),
                initial_derivative
                    ? elementwise::make_index(*initial_derivative)
                    : elementwise::Index{}
            );
        }

        template <typename T>
        void launch(
            core::ops::FractionalOp op,
            core::Tensor& out,
            const core::Tensor& input,
            const core::Tensor* initial_derivative,
            const core::Caputo& attrs,
            const double* weights,
            cudaStream_t stream
        ) {
            switch (op) {
                case core::ops::FractionalOp::caputo:
                    launch<T, core::ops::FractionalOp::caputo>(
                        out, input, initial_derivative, attrs, weights, stream
                    );
                    return;
                case core::ops::FractionalOp::caputo_adjoint:
                    launch<T, core::ops::FractionalOp::caputo_adjoint>(
                        out, input, nullptr, attrs, weights, stream
                    );
                    return;
                case core::ops::FractionalOp::caputo_initial:
                    launch<T, core::ops::FractionalOp::caputo_initial>(
                        out, input, nullptr, attrs, weights, stream
                    );
                    return;
                case core::ops::FractionalOp::caputo_initial_adjoint:
                    launch<T, core::ops::FractionalOp::caputo_initial_adjoint>(
                        out, input, nullptr, attrs, weights, stream
                    );
                    return;
            }

            ORMI_THROW(
                OperationError,
                "Unknown CUDA fractional differentiation operation."
            );
        }

        void drop_setup(void* state) noexcept {
            auto* prepared = static_cast<Prepared*>(state);
            prepared->memory->deallocate(
                prepared->workspace,
                prepared->workspace_bytes
            );
            prepared->memory->deallocate(
                prepared->weights,
                prepared->weights_bytes
            );
            delete prepared;
        }

        std::unique_ptr<Prepared> make_setup(
            core::ops::FractionalOp op,
            core::Tensor& out,
            const core::Caputo& attrs
        ) {
            auto& cuda_state = state(out.device());
            check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

            const auto& host_weights =
                op == core::ops::FractionalOp::caputo_adjoint
                    ? attrs.coefficients->adjoint
                    : attrs.coefficients->forward;
            auto result = std::make_unique<Prepared>(Prepared{
                .op = op,
                .memory = allocator(out.device()),
                .stream = cuda_state.stream(),
                .weights_bytes = host_weights.size() * sizeof(double),
                .workspace_bytes =
                    op == core::ops::FractionalOp::caputo &&
                    attrs.alpha > 1.0 &&
                    l2::staged(attrs.count)
                        ? out.bytes()
                        : 0
            });

            try {
                if (result->weights_bytes != 0) {
                    result->weights = static_cast<double*>(
                        result->memory->allocate(result->weights_bytes)
                    );
                }
                if (result->workspace_bytes != 0) {
                    result->workspace = result->memory->allocate(
                        result->workspace_bytes
                    );
                }
                if (result->weights_bytes != 0) {
                    check(
                        cudaMemcpyAsync(
                            result->weights,
                            host_weights.data(),
                            result->weights_bytes,
                            cudaMemcpyHostToDevice,
                            result->stream
                        ),
                        "cudaMemcpyAsyncCaputoWeights"
                    );
                }
            } catch (...) {
                result->memory->deallocate(
                    result->workspace,
                    result->workspace_bytes
                );
                result->memory->deallocate(
                    result->weights,
                    result->weights_bytes
                );
                throw;
            }

            return result;
        }

        void execute(
            Prepared& prepared,
            core::Tensor& out,
            const core::Tensor& input,
            const core::Tensor* initial_derivative,
            const core::Caputo& attrs
        ) {
            if (
                prepared.op == core::ops::FractionalOp::caputo &&
                attrs.alpha > 1.0 &&
                l2::staged(attrs.count)
            ) {
                l2::forward(
                    out,
                    input,
                    initial_derivative,
                    attrs,
                    prepared.weights,
                    prepared.workspace,
                    prepared.stream
                );
            } else {
                switch (out.dtype()) {
                    case core::DType::float64:
                        launch<double>(
                            prepared.op,
                            out,
                            input,
                            initial_derivative,
                            attrs,
                            prepared.weights,
                            prepared.stream
                        );
                        break;
                    case core::DType::float32:
                        launch<float>(
                            prepared.op,
                            out,
                            input,
                            initial_derivative,
                            attrs,
                            prepared.weights,
                            prepared.stream
                        );
                        break;
                    case core::DType::float16:
                    case core::DType::int32:
                    case core::DType::boolean:
                        ORMI_THROW(
                            DTypeError,
                            "CUDA fractional differentiation requires Float32 or Float64."
                        );
                }
            }

            check(cudaGetLastError(), "CUDA fractional kernel");
        }

        void run_setup(void* state, core::Tensor& out) {
            auto& prepared = *static_cast<Prepared*>(state);
            const auto& attrs = std::get<core::Caputo>(out.attrs());
            const auto* initial_derivative =
                prepared.op == core::ops::FractionalOp::caputo &&
                attrs.alpha > 1.0 &&
                out.parents().size() == 2
                    ? out.parents()[1].get()
                    : nullptr;
            execute(
                prepared,
                out,
                *out.parents()[0],
                initial_derivative,
                attrs
            );
        }
    }

    core::Setup prepare_fractional(
        core::ops::FractionalOp op,
        core::Tensor& out,
        const core::Caputo& attrs
    ) {
        if (out.numel() == 0) {
            return {};
        }

        auto prepared = make_setup(op, out, attrs);
        const auto buffers =
            static_cast<std::size_t>(prepared->weights != nullptr) +
            static_cast<std::size_t>(prepared->workspace != nullptr);
        if (
            prepared->weights_bytes >
            std::numeric_limits<std::size_t>::max() -
                prepared->workspace_bytes
        ) {
            drop_setup(prepared.release());
            ORMI_THROW(MemoryError, "CUDA fractional setup size overflow.");
        }
        const auto bytes =
            prepared->weights_bytes + prepared->workspace_bytes;
        return core::Setup(
            prepared.release(),
            run_setup,
            drop_setup,
            buffers,
            bytes
        );
    }

    void fractional(
        core::ops::FractionalOp op,
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor* initial_derivative,
        const core::Caputo& attrs
    ) {
        if (out.numel() == 0) {
            return;
        }

        auto prepared = make_setup(op, out, attrs);
        auto* state = prepared.get();
        core::Setup owner(
            prepared.release(),
            run_setup,
            drop_setup
        );
        execute(
            *state,
            out,
            input,
            initial_derivative,
            attrs
        );
    }
}
