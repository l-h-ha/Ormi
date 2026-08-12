#include <cstddef>
#include <limits>

#include <ormi/core/attrs.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/executor.h"
#include "backends/cpu/fractional/l1.h"
#include "backends/cpu/index.h"

namespace ormi::backends::cpu::caputo {
    namespace {
        template <typename T>
        struct Job {
            core::ops::FractionalOp op;
            core::Tensor& out;
            const core::Tensor& input;
            const core::Caputo& attrs;
            std::size_t axis_stride;
        };

        template <typename T>
        double adjoint(
            const Job<T>& job,
            const T* input,
            std::size_t base,
            std::size_t coordinate,
            std::size_t input_axis_stride
        ) {
            if (coordinate == 0) {
                if (job.attrs.count == 1) {
                    return 0.0;
                }

                double value = 0.0;
                auto weight = job.attrs.coefficients->adjoint[0];
                for (std::size_t n = 1; n < job.attrs.count; ++n) {
                    value -= weight * static_cast<double>(
                        input[base + n * input_axis_stride]
                    );

                    if (n < job.attrs.count - 1) {
                        weight += job.attrs.coefficients->adjoint[n];
                    }
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
        void run_range(
            void* context,
            [[maybe_unused]] std::size_t slot,
            std::size_t begin,
            std::size_t end
        ) {
            auto& job = *static_cast<Job<T>*>(context);
            auto* out = static_cast<T*>(job.out.data());
            const auto* input = static_cast<const T*>(job.input.data());
            const auto input_axis_stride = static_cast<std::size_t>(
                job.input.strides()[static_cast<std::size_t>(job.attrs.axis)]
            );

            for (std::size_t linear = begin; linear < end; ++linear) {
                const auto coordinate =
                    linear / job.axis_stride % job.attrs.count;
                const auto input_offset = storage_offset(linear, job.input);
                const auto base =
                    input_offset - coordinate * input_axis_stride;
                double value = 0.0;

                if (job.op == core::ops::FractionalOp::caputo) {
                    for (std::size_t k = 0; k < coordinate; ++k) {
                        const auto upper =
                            base + (coordinate - k) * input_axis_stride;
                        const auto lower = upper - input_axis_stride;
                        value += job.attrs.coefficients->forward[k] * (
                            static_cast<double>(input[upper]) -
                            static_cast<double>(input[lower])
                        );
                    }
                } else {
                    value = adjoint(
                        job,
                        input,
                        base,
                        coordinate,
                        input_axis_stride
                    );
                }
                out[linear] = static_cast<T>(value);
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
            const core::Caputo& attrs
        ) {
            if (out.numel() == 0) {
                return;
            }

            Job<T> job{
                op,
                out,
                input,
                attrs,
                static_cast<std::size_t>(
                    out.strides()[static_cast<std::size_t>(attrs.axis)]
                )
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

    void l1(
        core::ops::FractionalOp op,
        core::Tensor& out,
        const core::Tensor& input,
        const core::Caputo& attrs
    ) {
        switch (out.dtype()) {
            case core::DType::float64:
                apply<double>(op, out, input, attrs);
                return;
            case core::DType::float32:
                apply<float>(op, out, input, attrs);
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
