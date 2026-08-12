#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/index.h"
#include "backends/cpu/reduction/sum.h"

#if defined(ORMI_HAS_AVX2_KERNELS)
#include "backends/cpu/reduction/avx2.h"
#include "platform/platform.h"
#endif

namespace ormi::backends::cpu {
    namespace {
#if defined(ORMI_HAS_AVX2_KERNELS)
        constexpr std::size_t avx2_sum_elements = 1ull << 12;
#endif

        struct Slice {
            std::size_t outer;
            std::size_t reduced;
            std::size_t inner;
        };

        std::optional<Slice> contiguous_slice(
            const core::Tensor& out,
            const core::Tensor& input,
            std::span<const std::int64_t> axes
        ) {
            if (
                axes.empty() ||
                !out.is_contiguous() ||
                !input.is_contiguous()
            ) {
                return std::nullopt;
            }

            for (std::size_t i = 1; i < axes.size(); ++i) {
                if (axes[i] != axes[i - 1] + 1) {
                    return std::nullopt;
                }
            }

            const auto first = static_cast<std::size_t>(axes.front());
            const auto last = static_cast<std::size_t>(axes.back());
            Slice slice{1, 1, 1};

            for (std::size_t i = 0; i < first; ++i) {
                slice.outer *= static_cast<std::size_t>(input.shape()[i]);
            }
            for (std::size_t i = first; i <= last; ++i) {
                slice.reduced *= static_cast<std::size_t>(input.shape()[i]);
            }
            for (std::size_t i = last + 1; i < input.rank(); ++i) {
                slice.inner *= static_cast<std::size_t>(input.shape()[i]);
            }
            return slice;
        }

        template <typename Input, typename Sum>
        void contiguous_sum(
            Sum* sums,
            const Input* input,
            Slice slice
        ) {
            if (slice.inner == 1) {
                for (std::size_t outer = 0; outer < slice.outer; ++outer) {
                    Sum total{};
                    const auto* values = input + outer * slice.reduced;
                    for (
                        std::size_t reduced = 0;
                        reduced < slice.reduced;
                        ++reduced
                    ) {
                        total += static_cast<Sum>(values[reduced]);
                    }
                    sums[outer] = total;
                }
                return;
            }

            std::fill_n(sums, slice.outer * slice.inner, Sum{});
            for (std::size_t outer = 0; outer < slice.outer; ++outer) {
                auto* totals = sums + outer * slice.inner;
                const auto* values =
                    input + outer * slice.reduced * slice.inner;

                for (
                    std::size_t reduced = 0;
                    reduced < slice.reduced;
                    ++reduced
                ) {
                    const auto* row = values + reduced * slice.inner;
                    for (std::size_t inner = 0; inner < slice.inner; ++inner) {
                        totals[inner] += static_cast<Sum>(row[inner]);
                    }
                }
            }
        }

        template <typename T>
        void zero(core::Tensor& out) {
            auto* data = static_cast<T*>(out.data());
            const auto elements = out.numel();

            for (std::size_t i = 0; i < elements; ++i) {
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
            bool keepdims,
            T scale
        ) {
            const auto output_elements = out.numel();
            const auto input_elements = input.numel();
            if (const auto slice = contiguous_slice(out, input, axes)) {
                auto* out_data = static_cast<T*>(out.data());
                bool summed = false;

#if defined(ORMI_HAS_AVX2_KERNELS)
                if constexpr (std::is_same_v<T, float>) {
                    if (
                        slice->inner >= 8 &&
                        (
                            slice->reduced > avx2_sum_elements / slice->inner ||
                            slice->reduced * slice->inner >= avx2_sum_elements
                        )
                    ) {
                        static const bool has_avx2 = platform::supports_avx2();
                        if (has_avx2) {
                            const auto* input_data =
                                static_cast<const float*>(input.data());
                            for (
                                std::size_t outer = 0;
                                outer < slice->outer;
                                ++outer
                            ) {
                                avx2::sum_rows(
                                    out_data + outer * slice->inner,
                                    input_data +
                                        outer * slice->reduced * slice->inner,
                                    slice->reduced,
                                    slice->inner
                                );
                            }
                            summed = true;
                        }
                    }
                }
#endif

                if (!summed) {
                    contiguous_sum(
                        out_data,
                        static_cast<const T*>(input.data()),
                        *slice
                    );
                }

                if (scale != T{1}) {
                    for (std::size_t i = 0; i < output_elements; ++i) {
                        out_data[i] *= scale;
                    }
                }
                return;
            }

            zero<T>(out);

            auto* out_data = static_cast<T*>(out.data());
            const auto* input_data = static_cast<const T*>(input.data());
            const auto reduced = reduced_axes(input.rank(), axes);

            for (std::size_t i = 0; i < input_elements; ++i) {
                const auto output = reduced_offset(i, input, out, reduced, keepdims);
                out_data[output] += input_data[storage_offset(i, input)];
            }

            if (scale != T{1}) {
                for (std::size_t i = 0; i < output_elements; ++i) {
                    const auto offset = storage_offset(i, out);
                    out_data[offset] *= scale;
                }
            }
        }

        void reduce_float16(
            core::Tensor& out,
            const core::Tensor& input,
            std::span<const std::int64_t> axes,
            bool keepdims,
            float scale
        ) {
            auto* out_data = static_cast<core::Float16*>(out.data());
            const auto* input_data =
                static_cast<const core::Float16*>(input.data());
            const auto output_elements = out.numel();
            const auto input_elements = input.numel();
            std::vector<float> sums(output_elements, 0.0f);

            if (const auto slice = contiguous_slice(out, input, axes)) {
                contiguous_sum(sums.data(), input_data, *slice);
                for (std::size_t i = 0; i < output_elements; ++i) {
                    out_data[i] = core::Float16(sums[i] * scale);
                }
                return;
            }

            const auto reduced = reduced_axes(input.rank(), axes);

            for (std::size_t i = 0; i < input_elements; ++i) {
                const auto output = reduced_offset(i, input, out, reduced, keepdims);
                sums[output] += static_cast<float>(input_data[storage_offset(i, input)]);
            }

            for (std::size_t i = 0; i < output_elements; ++i) {
                out_data[storage_offset(i, out)] = core::Float16(sums[i] * scale);
            }
        }

        std::size_t reduced_elements(
            const core::Tensor& input,
            std::span<const std::int64_t> axes
        ) {
            std::size_t result = 1;

            for (const auto axis : axes) {
                result *= static_cast<std::size_t>(
                    input.shape()[static_cast<std::size_t>(axis)]
                );
            }
            return result;
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
                reduce_typed<double>(out, input, axes, keepdims, 1.0);
                return;
            case core::DType::float32:
                reduce_typed<float>(out, input, axes, keepdims, 1.0f);
                return;
            case core::DType::float16:
                reduce_float16(out, input, axes, keepdims, 1.0f);
                return;
            case core::DType::int32:
                reduce_typed<std::int32_t>(out, input, axes, keepdims, 1);
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CPU sum does not support Bool.");
        }

        ORMI_THROW(DTypeError, "CPU sum cannot handle an unknown dtype.");
    }

    void reduce_mean(
        core::Tensor& out,
        const core::Tensor& input,
        std::span<const std::int64_t> axes,
        bool keepdims
    ) {
        const auto count = reduced_elements(input, axes);

        switch (out.dtype()) {
            case core::DType::float64:
                reduce_typed<double>(
                    out,
                    input,
                    axes,
                    keepdims,
                    1.0 / static_cast<double>(count)
                );
                return;
            case core::DType::float32:
                reduce_typed<float>(
                    out,
                    input,
                    axes,
                    keepdims,
                    1.0f / static_cast<float>(count)
                );
                return;
            case core::DType::float16:
                reduce_float16(
                    out,
                    input,
                    axes,
                    keepdims,
                    1.0f / static_cast<float>(count)
                );
                return;
            case core::DType::int32:
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CPU mean requires a floating-point dtype.");
        }

        ORMI_THROW(DTypeError, "CPU mean cannot handle an unknown dtype.");
    }
}
