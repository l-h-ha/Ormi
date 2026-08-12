#include <cstddef>
#include <limits>

#include <ormi/core/attrs.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/executor.h"
#include "backends/cpu/fractional/l2.h"
#include "backends/cpu/index.h"

namespace ormi::backends::cpu::caputo {
    namespace {
        template <typename T>
        struct Job {
            core::ops::FractionalOp op;
            core::Tensor& out;
            const core::Tensor& input;
            const core::Tensor* initial_derivative;
            const core::Caputo& attrs;
            std::size_t axis_stride;
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
        double history_adjoint(
            const Job<T>& job,
            const T* input,
            std::size_t base,
            std::size_t coordinate,
            std::size_t input_axis_stride
        ) {
            if (coordinate == 0) {
                double value = 0.0;
                for (std::size_t n = 1; n < job.attrs.count; ++n) {
                    auto weight = -2.0 * job.attrs.coefficients->forward[n - 1];
                    if (n >= 2) {
                        weight += job.attrs.coefficients->forward[n - 2];
                    }
                    value += weight * static_cast<double>(
                        input[base + n * input_axis_stride]
                    );
                }
                return value;
            }
            if (coordinate == 1) {
                double value = 0.0;
                for (std::size_t n = 1; n < job.attrs.count; ++n) {
                    auto weight = 2.0 * job.attrs.coefficients->forward[n - 1];
                    if (n >= 2) {
                        weight -= 2.0 * job.attrs.coefficients->forward[n - 2];
                    }
                    if (n >= 3) {
                        weight += job.attrs.coefficients->forward[n - 3];
                    }
                    value += weight * static_cast<double>(
                        input[base + n * input_axis_stride]
                    );
                }
                return value;
            }

            auto value = job.attrs.coefficients->adjoint[0] *
                static_cast<double>(
                    input[base + coordinate * input_axis_stride]
                );
            for (
                std::size_t n = coordinate + 1;
                n < job.attrs.count;
                ++n
            ) {
                value += job.attrs.coefficients->adjoint[n - coordinate] *
                    static_cast<double>(
                        input[base + n * input_axis_stride]
                    );
            }
            return value;
        }

        template <typename T>
        double value(
            const Job<T>& job,
            const T* input,
            std::size_t linear,
            std::size_t base,
            std::size_t coordinate,
            std::size_t input_axis_stride
        ) {
            const T* initial = nullptr;
            std::size_t initial_offset = 0;
            if (job.initial_derivative) {
                initial = static_cast<const T*>(job.initial_derivative->data());
                const auto outer =
                    linear / (job.attrs.count * job.axis_stride);
                const auto inner = linear % job.axis_stride;
                initial_offset = storage_offset(
                    outer * job.axis_stride + inner,
                    *job.initial_derivative
                );
            }

            double result = 0.0;
            for (std::size_t k = 0; k < coordinate; ++k) {
                const auto difference = coordinate - 1 - k;
                double second_difference = 0.0;

                if (difference == 0) {
                    second_difference = 2.0 * (
                        static_cast<double>(
                            input[base + input_axis_stride]
                        ) -
                        static_cast<double>(input[base])
                    );
                    if (initial) {
                        second_difference -= 2.0 * job.attrs.spacing *
                            static_cast<double>(initial[initial_offset]);
                    }
                } else {
                    const auto center = base + difference * input_axis_stride;
                    second_difference =
                        static_cast<double>(input[center + input_axis_stride]) -
                        2.0 * static_cast<double>(input[center]) +
                        static_cast<double>(input[center - input_axis_stride]);
                }
                result += job.attrs.coefficients->forward[k] * second_difference;
            }
            return result;
        }

        template <typename T>
        void initial_adjoint_range(
            const Job<T>& job,
            T* out,
            const T* input,
            std::size_t begin,
            std::size_t end
        ) {
            for (std::size_t linear = begin; linear < end; ++linear) {
                const auto outer = linear / job.axis_stride;
                const auto inner = linear % job.axis_stride;
                const auto logical_base =
                    outer * job.attrs.count * job.axis_stride + inner;
                double result = 0.0;

                for (std::size_t n = 1; n < job.attrs.count; ++n) {
                    result -= 2.0 * job.attrs.spacing *
                        job.attrs.coefficients->forward[n - 1] *
                        static_cast<double>(input[storage_offset(
                            logical_base + n * job.axis_stride,
                            job.input
                        )]);
                }
                out[linear] = static_cast<T>(result);
            }
        }

        template <typename T>
        void run_range(
            void* context,
            [[maybe_unused]] std::size_t slot,
            std::size_t begin,
            std::size_t end
        ) {
            auto& job = *static_cast<Job<T>*>(context);
            auto* out = static_cast<T*>(job.out.data());
            const auto* input = static_cast<const T*>(job.input.data());

            if (job.op == core::ops::FractionalOp::caputo_initial_adjoint) {
                initial_adjoint_range(job, out, input, begin, end);
                return;
            }

            const auto input_axis_stride =
                job.op == core::ops::FractionalOp::caputo_initial
                    ? 0
                    : static_cast<std::size_t>(
                        job.input.strides()[
                            static_cast<std::size_t>(job.attrs.axis)
                        ]
                    );

            for (std::size_t linear = begin; linear < end; ++linear) {
                const auto coordinate =
                    linear / job.axis_stride % job.attrs.count;

                if (job.op == core::ops::FractionalOp::caputo_initial) {
                    if (coordinate == 0) {
                        out[linear] = T{};
                        continue;
                    }

                    const auto outer =
                        linear / (job.attrs.count * job.axis_stride);
                    const auto inner = linear % job.axis_stride;
                    const auto offset = storage_offset(
                        outer * job.axis_stride + inner,
                        job.input
                    );
                    out[linear] = static_cast<T>(
                        -2.0 * job.attrs.spacing *
                        job.attrs.coefficients->forward[coordinate - 1] *
                        static_cast<double>(input[offset])
                    );
                    continue;
                }

                const auto input_offset = storage_offset(linear, job.input);
                const auto base =
                    input_offset - coordinate * input_axis_stride;
                const auto result =
                    job.op == core::ops::FractionalOp::caputo
                        ? value(
                            job,
                            input,
                            linear,
                            base,
                            coordinate,
                            input_axis_stride
                        )
                        : history_adjoint(
                            job,
                            input,
                            base,
                            coordinate,
                            input_axis_stride
                        );
                out[linear] = static_cast<T>(result);
            }
        }

        std::size_t work(std::size_t elements, std::size_t count) {
            if (
                count != 0 &&
                elements > std::numeric_limits<std::size_t>::max() / count
            ) {
                return std::numeric_limits<std::size_t>::max();
            }
            return elements * count;
        }

        template <typename T>
        void apply(
            core::ops::FractionalOp op,
            core::Tensor& out,
            const core::Tensor& input,
            const core::Tensor* initial_derivative,
            const core::Caputo& attrs
        ) {
            if (out.numel() == 0) {
                return;
            }

            const auto axis_stride =
                op == core::ops::FractionalOp::caputo_initial_adjoint
                    ? initial_axis_stride(
                        out,
                        static_cast<std::size_t>(attrs.axis)
                    )
                    : static_cast<std::size_t>(
                        out.strides()[static_cast<std::size_t>(attrs.axis)]
                    );
            Job<T> job{
                op,
                out,
                input,
                initial_derivative,
                attrs,
                axis_stride
            };
            executor::run(
                out.numel(),
                1,
                work(out.numel(), attrs.count),
                run_range<T>,
                &job
            );
        }
    }

    void l2(
        core::ops::FractionalOp op,
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor* initial_derivative,
        const core::Caputo& attrs
    ) {
        switch (out.dtype()) {
            case core::DType::float64:
                apply<double>(op, out, input, initial_derivative, attrs);
                return;
            case core::DType::float32:
                apply<float>(op, out, input, initial_derivative, attrs);
                return;
            case core::DType::float16:
            case core::DType::int32:
            case core::DType::boolean:
                ORMI_THROW(
                    DTypeError,
                    "CPU fractional differentiation requires Float32 or Float64."
                );
        }

        ORMI_THROW(
            DTypeError,
            "CPU fractional differentiation received an unknown dtype."
        );
    }
}
