#include <cstddef>

#include <cuda_runtime_api.h>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/elementwise/index.cuh"
#include "backends/cuda/fractional/l2.h"

namespace ormi::backends::cuda::l2 {
    namespace {
        constexpr std::size_t staging_min_history = 64;

        template <typename T>
        __global__ void differences_kernel(
            T* differences,
            const T* input,
            const T* initial_derivative,
            std::size_t elements,
            std::size_t count,
            std::size_t axis_stride,
            std::size_t input_axis_stride,
            double spacing,
            elementwise::Index input_index,
            elementwise::Index initial_index
        ) {
            std::size_t linear =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                const auto coordinate = linear / axis_stride % count;
                if (coordinate + 1 >= count) {
                    differences[linear] = T{};
                    continue;
                }

                const auto input_offset =
                    elementwise::storage_offset(linear, input_index);
                if (coordinate == 0) {
                    auto value = static_cast<T>(2) * (
                        input[input_offset + input_axis_stride] -
                        input[input_offset]
                    );
                    if (initial_derivative) {
                        const auto outer = linear / (count * axis_stride);
                        const auto inner = linear % axis_stride;
                        const auto initial_offset = elementwise::storage_offset(
                            outer * axis_stride + inner,
                            initial_index
                        );
                        value -= static_cast<T>(2.0 * spacing) *
                            initial_derivative[initial_offset];
                    }
                    differences[linear] = value;
                    continue;
                }

                differences[linear] =
                    input[input_offset + input_axis_stride] -
                    static_cast<T>(2) * input[input_offset] +
                    input[input_offset - input_axis_stride];
            }
        }

        template <typename T>
        __global__ void convolution_kernel(
            T* out,
            const T* differences,
            const double* weights,
            std::size_t elements,
            std::size_t count,
            std::size_t axis_stride
        ) {
            std::size_t linear =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                const auto coordinate = linear / axis_stride % count;
                const auto base = linear - coordinate * axis_stride;
                T value{};

                for (std::size_t k = 0; k < coordinate; ++k) {
                    value += static_cast<T>(weights[k]) * differences[
                        base + (coordinate - 1 - k) * axis_stride
                    ];
                }
                out[linear] = value;
            }
        }

        template <typename T>
        void launch(
            core::Tensor& out,
            const core::Tensor& input,
            const core::Tensor* initial_derivative,
            const core::Caputo& attrs,
            const double* weights,
            T* differences,
            cudaStream_t stream
        ) {
            const auto axis = static_cast<std::size_t>(attrs.axis);
            const auto axis_stride =
                static_cast<std::size_t>(out.strides()[axis]);
            const auto input_axis_stride =
                static_cast<std::size_t>(input.strides()[axis]);

            differences_kernel<T><<<
                elementwise::blocks(out.numel()),
                elementwise::threads,
                0,
                stream
            >>>(
                differences,
                static_cast<const T*>(input.data()),
                initial_derivative
                    ? static_cast<const T*>(initial_derivative->data())
                    : nullptr,
                out.numel(),
                attrs.count,
                axis_stride,
                input_axis_stride,
                attrs.spacing,
                elementwise::make_index(input),
                initial_derivative
                    ? elementwise::make_index(*initial_derivative)
                    : elementwise::Index{}
            );
            check(cudaGetLastError(), "CUDA Caputo L2 differences kernel");

            convolution_kernel<T><<<
                elementwise::blocks(out.numel()),
                elementwise::threads,
                0,
                stream
            >>>(
                static_cast<T*>(out.data()),
                differences,
                weights,
                out.numel(),
                attrs.count,
                axis_stride
            );
            check(cudaGetLastError(), "CUDA Caputo L2 convolution kernel");
        }
    }

    bool staged(std::size_t count) noexcept {
        return count >= staging_min_history;
    }

    void forward(
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor* initial_derivative,
        const core::Caputo& attrs,
        const double* weights,
        void* workspace,
        cudaStream_t stream
    ) {
        switch (out.dtype()) {
            case core::DType::float64:
                launch<double>(
                    out,
                    input,
                    initial_derivative,
                    attrs,
                    weights,
                    static_cast<double*>(workspace),
                    stream
                );
                return;
            case core::DType::float32:
                launch<float>(
                    out,
                    input,
                    initial_derivative,
                    attrs,
                    weights,
                    static_cast<float*>(workspace),
                    stream
                );
                return;
            case core::DType::float16:
            case core::DType::int32:
            case core::DType::boolean:
                ORMI_THROW(
                    DTypeError,
                    "CUDA fractional differentiation requires Float32 or Float64."
                );
        }
    }
}
