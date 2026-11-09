#include <cmath>
#include <cstdint>
#include <type_traits>

#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/pointwise.h"
#include "backends/cpu/pointwise/unary.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename T>
        void pointwise_unary_typed(core::ops::UnaryOp op, core::Tensor& out, const core::Tensor& input) {
            switch (op) {
                case core::ops::UnaryOp::rsqrt:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return T(1) / std::sqrt(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU rsqrt requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::abs:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::fabs(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU abs requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::sign:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) {
                            return value != value ? value : T((value > T(0)) - (value < T(0)));
                        });
                    } else {
                        ORMI_THROW(DTypeError, "CPU sign requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::positive:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) {
                            return value != value ? value : T(value > T(0));
                        });
                    } else {
                        ORMI_THROW(DTypeError, "CPU positive requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::relu:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return value < T(0) ? T(0) : value; });
                    } else {
                        ORMI_THROW(DTypeError, "CPU relu requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::tanh:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::tanh(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU tanh requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::sigmoid:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) {
                            return value >= T(0) ? T(1) / (T(1) + std::exp(-value))
                                                 : std::exp(value) / (T(1) + std::exp(value));
                        });
                    } else {
                        ORMI_THROW(DTypeError, "CPU sigmoid requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::softplus:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) {
                            return value > T(0) ? value + std::log1p(std::exp(-value))
                                                : std::log1p(std::exp(value));
                        });
                    } else {
                        ORMI_THROW(DTypeError, "CPU softplus requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::erf:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::erf(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU erf requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::log1p:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::log1p(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU log1p requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::expm1:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::expm1(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU expm1 requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::neg:
                    unary<T>(out, input, [](T value) { return -value; });
                    return;
                case core::ops::UnaryOp::exp:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::exp(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU exp requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::log:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::log(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU log requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::sin:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::sin(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU sin requires a floating-point tensor.");
                    }
                    return;
                case core::ops::UnaryOp::cos:
                    if constexpr (std::is_floating_point_v<T>) {
                        unary<T>(out, input, [](T value) { return std::cos(value); });
                    } else {
                        ORMI_THROW(DTypeError, "CPU cos requires a floating-point tensor.");
                    }
                    return;
            }

            ORMI_THROW(OperationError, "Unknown CPU unary pointwise operation.");
        }

        void pointwise_unary_float16(core::ops::UnaryOp op, core::Tensor& out, const core::Tensor& input) {
            using core::Float16;

            switch (op) {
                case core::ops::UnaryOp::rsqrt:
                    unary<Float16>(out, input, [](Float16 value) {
                        return Float16(1.0f / std::sqrt(static_cast<float>(value)));
                    });
                    return;
                case core::ops::UnaryOp::abs:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(std::fabs(value));
                    });
                    return;
                case core::ops::UnaryOp::sign:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(value != value ? value : T((value > T(0)) - (value < T(0))));
                    });
                    return;
                case core::ops::UnaryOp::positive:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(value != value ? value : T(value > T(0)));
                    });
                    return;
                case core::ops::UnaryOp::relu:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(value < T(0) ? T(0) : value);
                    });
                    return;
                case core::ops::UnaryOp::tanh:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(std::tanh(value));
                    });
                    return;
                case core::ops::UnaryOp::sigmoid:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(
                            value >= T(0) ? T(1) / (T(1) + std::exp(-value))
                                          : std::exp(value) / (T(1) + std::exp(value))
                        );
                    });
                    return;
                case core::ops::UnaryOp::softplus:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(
                            value > T(0) ? value + std::log1p(std::exp(-value)) : std::log1p(std::exp(value))
                        );
                    });
                    return;
                case core::ops::UnaryOp::erf:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(std::erf(value));
                    });
                    return;
                case core::ops::UnaryOp::log1p:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(std::log1p(value));
                    });
                    return;
                case core::ops::UnaryOp::expm1:
                    unary<Float16>(out, input, [](Float16 stored) {
                        using T = float;
                        const float value = static_cast<float>(stored);
                        return Float16(std::expm1(value));
                    });
                    return;
                case core::ops::UnaryOp::neg:
                    unary<Float16>(out, input, [](Float16 value) { return -value; });
                    return;
                case core::ops::UnaryOp::exp:
                    unary<Float16>(out, input, [](Float16 value) {
                        return Float16(std::exp(static_cast<float>(value)));
                    });
                    return;
                case core::ops::UnaryOp::log:
                    unary<Float16>(out, input, [](Float16 value) {
                        return Float16(std::log(static_cast<float>(value)));
                    });
                    return;
                case core::ops::UnaryOp::sin:
                    unary<Float16>(out, input, [](Float16 value) {
                        return Float16(std::sin(static_cast<float>(value)));
                    });
                    return;
                case core::ops::UnaryOp::cos:
                    unary<Float16>(out, input, [](Float16 value) {
                        return Float16(std::cos(static_cast<float>(value)));
                    });
                    return;
            }

            ORMI_THROW(OperationError, "Unknown CPU unary pointwise operation.");
        }
    }

    void pointwise_unary(core::ops::UnaryOp op, core::Tensor& out, const core::Tensor& input) {
        switch (out.dtype()) {
            case core::DType::float64:
                pointwise_unary_typed<double>(op, out, input);
                return;
            case core::DType::float32:
                pointwise_unary_typed<float>(op, out, input);
                return;
            case core::DType::float16:
                pointwise_unary_float16(op, out, input);
                return;
            case core::DType::int32:
                pointwise_unary_typed<std::int32_t>(op, out, input);
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CPU unary pointwise operations do not support Bool.");
        }

        ORMI_THROW(DTypeError, "CPU unary pointwise operations cannot handle an unknown dtype.");
    }
}
