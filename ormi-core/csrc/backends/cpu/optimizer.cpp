#include <cstdint>
#include <type_traits>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/index.h"
#include "backends/cpu/optimizer.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename T>
        T updated(T parameter, T gradient, double step_size) {
            if constexpr (std::is_same_v<T, core::Float16>) {
                return core::Float16(
                    static_cast<float>(parameter) -
                    static_cast<float>(step_size) * static_cast<float>(gradient)
                );
            } else {
                return parameter - static_cast<T>(step_size) * gradient;
            }
        }

        template <typename T>
        void sgd_typed(
            core::Tensor& out,
            const core::Tensor& parameter,
            const core::Tensor& gradient,
            double step_size
        ) {
            auto* out_data = static_cast<T*>(out.data());
            const auto* parameter_data = static_cast<const T*>(parameter.data());
            const auto* gradient_data = static_cast<const T*>(gradient.data());

            if (out.is_contiguous() && parameter.is_contiguous() && gradient.is_contiguous()) {
                for (std::size_t i = 0; i < out.numel(); ++i) {
                    out_data[i] = updated(parameter_data[i], gradient_data[i], step_size);
                }
                return;
            }

            for (std::size_t i = 0; i < out.numel(); ++i) {
                const auto out_offset = storage_offset(i, out);
                const auto parameter_offset = storage_offset(i, parameter);
                const auto gradient_offset = storage_offset(i, gradient);
                out_data[out_offset] = updated(
                    parameter_data[parameter_offset],
                    gradient_data[gradient_offset],
                    step_size
                );
            }
        }
    }

    void sgd(
        core::Tensor& out,
        const core::Tensor& parameter,
        const core::Tensor& gradient,
        double step_size
    ) {
        switch (out.dtype()) {
            case core::DType::float64:
                sgd_typed<double>(out, parameter, gradient, step_size);
                return;
            case core::DType::float32:
                sgd_typed<float>(out, parameter, gradient, step_size);
                return;
            case core::DType::float16:
                sgd_typed<core::Float16>(out, parameter, gradient, step_size);
                return;
            case core::DType::int32:
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CPU SGD supports floating-point tensors only.");
        }

        ORMI_THROW(DTypeError, "CPU SGD cannot handle an unknown dtype.");
    }
}
