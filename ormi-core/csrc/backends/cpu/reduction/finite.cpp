#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/index.h"
#include "backends/cpu/reduction/finite.h"

namespace ormi::backends::cpu {
    namespace {
        static_assert(sizeof(float) == sizeof(std::uint32_t));
        static_assert(sizeof(double) == sizeof(std::uint64_t));
        static_assert(std::numeric_limits<float>::is_iec559);
        static_assert(std::numeric_limits<double>::is_iec559);

        template <typename T>
        bool finite(T value) {
            if constexpr (std::is_same_v<T, double>) {
                return (std::bit_cast<std::uint64_t>(value) & 0x7ff0000000000000ull) !=
                    0x7ff0000000000000ull;
            } else if constexpr (std::is_same_v<T, float>) {
                return (std::bit_cast<std::uint32_t>(value) & 0x7f800000u) != 0x7f800000u;
            } else {
                return (value.bits() & 0x7c00u) != 0x7c00u;
            }
        }

        template <typename T>
        bool tensor_all_finite(const core::Tensor& input) {
            const auto* data = static_cast<const T*>(input.data());
            const auto elements = input.numel();

            if (input.is_contiguous()) {
                constexpr std::size_t block_size = 256;

                for (
                    std::size_t block = 0;
                    block < elements;
                    block += block_size
                ) {
                    const auto end = std::min(block + block_size, elements);
                    bool valid = true;

                    for (std::size_t i = block; i < end; ++i) {
                        valid &= finite(data[i]);
                    }

                    if (!valid) {
                        return false;
                    }
                }

                return true;
            }

            for (std::size_t i = 0; i < elements; ++i) {
                if (!finite(data[storage_offset(i, input)])) {
                    return false;
                }
            }

            return true;
        }

        bool tensor_all_finite(const core::Tensor& input) {
            switch (input.dtype()) {
                case core::DType::float64:
                    return tensor_all_finite<double>(input);
                case core::DType::float32:
                    return tensor_all_finite<float>(input);
                case core::DType::float16:
                    return tensor_all_finite<core::Float16>(input);
                case core::DType::int32:
                case core::DType::boolean:
                    ORMI_THROW(DTypeError, "CPU all-finite supports floating-point tensors only.");
            }

            ORMI_THROW(DTypeError, "CPU all-finite cannot handle an unknown dtype.");
        }
    }

    bool all_finite(std::span<const core::Tensor* const> inputs) {
        for (const auto* input : inputs) {
            if (!tensor_all_finite(*input)) {
                return false;
            }
        }

        return true;
    }
}
