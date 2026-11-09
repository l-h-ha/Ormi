#include <cstddef>
#include <cstdint>
#include <limits>

#include <ormi/core/attrs.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>

#include "backends/cpu/index.h"
#include "backends/cpu/indexing.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename T, bool scatter>
        void select(
            core::Tensor& out,
            const core::Tensor& input,
            const core::Tensor& indices,
            const core::Pick& attrs
        ) {
            const auto axis = static_cast<std::size_t>(attrs.axis);
            const auto& shape = attrs.shape;
            const auto extent = shape[axis];
            const auto strides = input.strides();
            const auto* labels = static_cast<const std::int32_t*>(indices.data());
            const auto* values = static_cast<const T*>(input.data());
            auto* output = static_cast<T*>(out.data());

            for (std::size_t linear = 0; linear < out.numel(); ++linear) {
                std::size_t remaining = linear;
                std::size_t value_offset = 0;
                std::size_t label_linear = 0;
                std::size_t label_stride = 1;
                std::int64_t class_coordinate = 0;

                for (std::size_t i = shape.rank(); i > 0; --i) {
                    const auto dimension = i - 1;
                    if (dimension == axis && !scatter) {
                        continue;
                    }

                    const auto size = static_cast<std::size_t>(shape[dimension]);
                    const auto coordinate = remaining % size;
                    remaining /= size;
                    if (dimension == axis) {
                        class_coordinate = static_cast<std::int64_t>(coordinate);
                        continue;
                    }

                    label_linear += coordinate * label_stride;
                    label_stride *= size;
                    const auto value_axis = scatter && dimension > axis ? dimension - 1 : dimension;
                    value_offset += coordinate * static_cast<std::size_t>(strides[value_axis]);
                }

                const auto label = labels[storage_offset(label_linear, indices)];
                if (attrs.ignore_index && label == *attrs.ignore_index) {
                    output[linear] = T(0.0f);
                    continue;
                }
                if (label < 0 || label >= extent) {
                    output[linear] = T(std::numeric_limits<float>::quiet_NaN());
                    continue;
                }

                if constexpr (scatter) {
                    output[linear] = label == class_coordinate ? values[value_offset] : T(0.0f);
                } else {
                    const auto class_offset =
                        static_cast<std::size_t>(label) * static_cast<std::size_t>(strides[axis]);
                    output[linear] = values[value_offset + class_offset];
                }
            }
        }

        template <bool scatter>
        void dispatch(
            core::Tensor& out,
            const core::Tensor& input,
            const core::Tensor& indices,
            const core::Pick& attrs
        ) {
            switch (out.dtype()) {
                case core::DType::float64:
                    select<double, scatter>(out, input, indices, attrs);
                    return;
                case core::DType::float32:
                    select<float, scatter>(out, input, indices, attrs);
                    return;
                case core::DType::float16:
                    select<core::Float16, scatter>(out, input, indices, attrs);
                    return;
                default:
                    ORMI_THROW(DTypeError, "Indexed selection requires floating-point input.");
            }
        }
    }

    void pick(
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor& indices,
        const core::Pick& attrs
    ) {
        dispatch<false>(out, input, indices, attrs);
    }

    void unpick(
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor& indices,
        const core::Pick& attrs
    ) {
        dispatch<true>(out, input, indices, attrs);
    }
}
