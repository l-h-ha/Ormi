#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/cast.h"
#include "backends/cpu/index.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename To, typename From>
        To convert(From value) {
            if constexpr (std::is_same_v<To, std::int32_t>) {
                const auto number = [&] {
                    if constexpr (std::is_same_v<From, core::Float16>) {
                        return static_cast<double>(static_cast<float>(value));
                    } else {
                        return static_cast<double>(value);
                    }
                }();

                if (
                    !std::isfinite(number) ||
                    number < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
                    number > static_cast<double>(std::numeric_limits<std::int32_t>::max())
                ) {
                    ORMI_THROW(DTypeError, "CPU cast to Int32 requires a finite value within the Int32 range.");
                }

                return static_cast<std::int32_t>(number);
            } else if constexpr (std::is_same_v<To, core::Float16>) {
                return core::Float16(static_cast<float>(value));
            } else if constexpr (std::is_same_v<From, core::Float16>) {
                return static_cast<To>(static_cast<float>(value));
            } else {
                return static_cast<To>(value);
            }
        }

        template <typename To, typename From>
        void cast_typed(core::Tensor& out, const core::Tensor& input) {
            auto* out_data = static_cast<To*>(out.data());
            const auto* input_data = static_cast<const From*>(input.data());
            const auto elements = out.numel();

            if (input.is_contiguous()) {
                for (std::size_t i = 0; i < elements; ++i) {
                    out_data[i] = convert<To>(input_data[i]);
                }
                return;
            }

            for (std::size_t i = 0; i < elements; ++i) {
                out_data[i] = convert<To>(input_data[storage_offset(i, input)]);
            }
        }

        template <typename From>
        void cast_from(core::Tensor& out, const core::Tensor& input) {
            switch (out.dtype()) {
                case core::DType::float64:
                    cast_typed<double, From>(out, input);
                    return;
                case core::DType::float32:
                    cast_typed<float, From>(out, input);
                    return;
                case core::DType::float16:
                    cast_typed<core::Float16, From>(out, input);
                    return;
                case core::DType::int32:
                    cast_typed<std::int32_t, From>(out, input);
                    return;
                case core::DType::boolean:
                    cast_typed<bool, From>(out, input);
                    return;
            }

            ORMI_THROW(DTypeError, "CPU cast cannot handle an unknown output dtype.");
        }
    }

    void cast(core::Tensor& out, const core::Tensor& input) {
        switch (input.dtype()) {
            case core::DType::float64:
                cast_from<double>(out, input);
                return;
            case core::DType::float32:
                cast_from<float>(out, input);
                return;
            case core::DType::float16:
                cast_from<core::Float16>(out, input);
                return;
            case core::DType::int32:
                cast_from<std::int32_t>(out, input);
                return;
            case core::DType::boolean:
                cast_from<bool>(out, input);
                return;
        }

        ORMI_THROW(DTypeError, "CPU cast cannot handle an unknown input dtype.");
    }
}
