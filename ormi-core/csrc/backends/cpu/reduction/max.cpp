#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/reduction/max.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename T, typename Value>
        void maximum(core::Tensor& out, const core::Tensor& input, std::span<const std::int64_t> axes) {
            std::vector<bool> reduced(input.rank(), false);
            std::size_t count = 1;
            for (const auto axis : axes) {
                reduced[static_cast<std::size_t>(axis)] = true;
                count *= static_cast<std::size_t>(input.shape()[static_cast<std::size_t>(axis)]);
            }
            bool contiguous = input.is_contiguous();
            for (std::size_t i = 0; i < axes.size(); ++i) {
                contiguous =
                    contiguous && static_cast<std::size_t>(axes[i]) == input.rank() - axes.size() + i;
            }
            auto* destination = static_cast<T*>(out.data());
            const auto* source = static_cast<const T*>(input.data());
            const auto strides = input.strides();
            for (std::size_t output = 0; output < out.numel(); ++output) {
                if (contiguous) {
                    Value value = -std::numeric_limits<Value>::infinity();
                    for (std::size_t i = 0; i < count; ++i) {
                        const Value candidate = static_cast<Value>(source[output * count + i]);
                        if (candidate > value || std::isnan(candidate)) {
                            value = candidate;
                        }
                    }
                    destination[output] = T(value);
                    continue;
                }
                std::size_t linear = output, base = 0;
                for (std::size_t i = input.rank(); i > 0; --i) {
                    const auto axis = i - 1;
                    if (reduced[axis]) {
                        continue;
                    }
                    const auto extent = static_cast<std::size_t>(input.shape()[axis]);
                    base += (linear % extent) * static_cast<std::size_t>(strides[axis]);
                    linear /= extent;
                }
                Value value = -std::numeric_limits<Value>::infinity();
                for (std::size_t element = 0; element < count; ++element) {
                    std::size_t index = element, offset = base;
                    for (std::size_t i = axes.size(); i > 0; --i) {
                        const auto axis = static_cast<std::size_t>(axes[i - 1]);
                        const auto extent = static_cast<std::size_t>(input.shape()[axis]);
                        offset += (index % extent) * static_cast<std::size_t>(strides[axis]);
                        index /= extent;
                    }
                    const Value candidate = static_cast<Value>(source[offset]);
                    if (candidate > value || std::isnan(candidate)) {
                        value = candidate;
                    }
                }
                destination[output] = T(value);
            }
        }
    }

    void reduce_max(core::Tensor& out, const core::Tensor& input, std::span<const std::int64_t> axes) {
        switch (out.dtype()) {
            case core::DType::float64:
                maximum<double, double>(out, input, axes);
                return;
            case core::DType::float32:
                maximum<float, float>(out, input, axes);
                return;
            case core::DType::float16:
                maximum<core::Float16, float>(out, input, axes);
                return;
            default:
                ORMI_THROW(DTypeError, "CPU max requires floating-point tensors.");
        }
    }
}
