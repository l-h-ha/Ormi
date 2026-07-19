#include <cstdint>
#include <cmath>
#include <type_traits>

#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/pointwise.h"
#include "backends/cpu/pointwise/unary.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename T>
        void pointwise_unary_typed(core::ops::UnaryOp op, core::Tensor& out, const core::Tensor& t) {
            switch (op) {
                case core::ops::UnaryOp::neg:
                    unary<T>(out, t, [](T value) { return -value; });
                    return;
                case core::ops::UnaryOp::exp:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, t, [](T value) { return std::exp(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU exp currently supports Float32 only.");
                    }
                    return;
                case core::ops::UnaryOp::log:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, t, [](T value) { return std::log(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU log currently supports Float32 only.");
                    }
                    return;
                case core::ops::UnaryOp::sin:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, t, [](T value) { return std::sin(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU sin currently supports Float32 only.");
                    }
                    return;
                case core::ops::UnaryOp::cos:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, t, [](T value) { return std::cos(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU cos currently supports Float32 only.");
                    }
                    return;
            }

            ORMI_THROW(OperationError, "Unknown CPU unary pointwise operation.");
        }

        void pointwise_unary_float16(core::ops::UnaryOp op, core::Tensor& out, const core::Tensor& t) {
            using core::Float16;

            switch (op) {
                case core::ops::UnaryOp::neg:
                    unary<Float16>(out, t, [](Float16 value) { return -value; });
                    return;
                case core::ops::UnaryOp::exp:
                    unary<Float16>(out, t, [](Float16 value) {
                        return Float16(std::exp(static_cast<float>(value)));
                    });
                    return;
                case core::ops::UnaryOp::log:
                    unary<Float16>(out, t, [](Float16 value) {
                        return Float16(std::log(static_cast<float>(value)));
                    });
                    return;
                case core::ops::UnaryOp::sin:
                    unary<Float16>(out, t, [](Float16 value) {
                        return Float16(std::sin(static_cast<float>(value)));
                    });
                    return;
                case core::ops::UnaryOp::cos:
                    unary<Float16>(out, t, [](Float16 value) {
                        return Float16(std::cos(static_cast<float>(value)));
                    });
                    return;
            }

            ORMI_THROW(OperationError, "Unknown CPU unary pointwise operation.");
        }
    }

    void pointwise_unary(core::ops::UnaryOp op, core::Tensor& out, const core::Tensor& t) {
        switch (out.dtype()) {
            case core::DType::float64:
                pointwise_unary_typed<double>(op, out, t);
                return;
            case core::DType::float32:
                pointwise_unary_typed<float>(op, out, t);
                return;
            case core::DType::float16:
                pointwise_unary_float16(op, out, t);
                return;
            case core::DType::int32:
                pointwise_unary_typed<std::int32_t>(op, out, t);
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CPU unary pointwise operations do not support Bool.");
        }

        ORMI_THROW(DTypeError, "CPU unary pointwise operations cannot handle an unknown dtype.");
    }
}
