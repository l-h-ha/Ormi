#include <cstdint>
#include <span>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/index.h"
#include "backends/cpu/reduction/sum.h"

namespace ormi::backends::cpu {
    namespace {
        template <typename T>
        void zero(core::Tensor& out) {
            auto* data = static_cast<T*>(out.data());

            for (std::size_t i = 0; i < out.numel(); ++i) {
                data[storage_offset(i, out)] = T{};
            }
        }

        std::size_t reduced_offset(
            std::size_t linear,
            const core::Tensor& input,
            const core::Tensor& out,
            std::span<const unsigned char> reduced_axes,
            bool keepdims
        ) {
            const auto& input_dims = input.shape().dims();
            const auto out_strides = out.strides();
            std::size_t offset = 0;
            std::size_t out_axis = out.rank();

            for (std::size_t axis = input.rank(); axis > 0; --axis) {
                const auto input_axis = axis - 1;
                const auto coordinate = linear % static_cast<std::size_t>(input_dims[input_axis]);
                linear /= static_cast<std::size_t>(input_dims[input_axis]);

                if (reduced_axes[input_axis] != 0) {
                    if (keepdims) {
                        --out_axis;
                    }
                    continue;
                }

                --out_axis;
                offset += coordinate * static_cast<std::size_t>(out_strides[out_axis]);
            }

            return offset;
        }

        std::vector<unsigned char> reduced_axes(
            std::size_t rank,
            std::span<const std::int64_t> axes
        ) {
            std::vector<unsigned char> result(rank, 0);

            for (const auto axis : axes) {
                result[static_cast<std::size_t>(axis)] = 1;
            }

            return result;
        }

        template <typename T>
        void reduce_typed(
            core::Tensor& out,
            const core::Tensor& input,
            std::span<const std::int64_t> axes,
            bool keepdims
        ) {
            zero<T>(out);

            auto* out_data = static_cast<T*>(out.data());
            const auto* input_data = static_cast<const T*>(input.data());
            const auto reduced = reduced_axes(input.rank(), axes);

            for (std::size_t i = 0; i < input.numel(); ++i) {
                const auto output = reduced_offset(i, input, out, reduced, keepdims);
                out_data[output] += input_data[storage_offset(i, input)];
            }
        }

        void reduce_float16(
            core::Tensor& out,
            const core::Tensor& input,
            std::span<const std::int64_t> axes,
            bool keepdims
        ) {
            auto* out_data = static_cast<core::Float16*>(out.data());
            const auto* input_data = static_cast<const core::Float16*>(input.data());
            std::vector<float> sums(out.numel(), 0.0f);
            const auto reduced = reduced_axes(input.rank(), axes);

            for (std::size_t i = 0; i < input.numel(); ++i) {
                const auto output = reduced_offset(i, input, out, reduced, keepdims);
                sums[output] += static_cast<float>(input_data[storage_offset(i, input)]);
            }

            for (std::size_t i = 0; i < out.numel(); ++i) {
                out_data[storage_offset(i, out)] = core::Float16(sums[i]);
            }
        }
    }

    void reduce_sum(
        core::Tensor& out,
        const core::Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    ) {
        switch (out.dtype()) {
            case core::DType::float64:
                reduce_typed<double>(out, input, axes, keepdims);
                return;
            case core::DType::float32:
                reduce_typed<float>(out, input, axes, keepdims);
                return;
            case core::DType::float16:
                reduce_float16(out, input, axes, keepdims);
                return;
            case core::DType::int32:
                reduce_typed<std::int32_t>(out, input, axes, keepdims);
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CPU sum does not support Bool.");
        }

        ORMI_THROW(DTypeError, "CPU sum cannot handle an unknown dtype.");
    }
}
