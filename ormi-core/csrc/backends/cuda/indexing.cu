#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ormi/core/attrs.h>
#include <ormi/core/error.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/elementwise/index.cuh"
#include "backends/cuda/indexing.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        struct Index {
            std::size_t rank;
            std::size_t axis;
            std::size_t dims[64]{};
            std::size_t value_strides[64]{};
            std::size_t label_strides[64]{};
            std::int32_t ignore_index;
            bool has_ignore_index;
        };

        template <typename T>
        __device__ T stored(float value) {
            if constexpr (std::is_same_v<T, __half>) {
                return __float2half_rn(value);
            } else {
                return static_cast<T>(value);
            }
        }

        template <typename T, bool scatter>
        __global__ void select_kernel(
            T* out,
            const T* input,
            const std::int32_t* labels,
            std::size_t count,
            Index index
        ) {
            const auto step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
            for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 linear < count;
                 linear += step) {
                std::size_t remaining = linear;
                std::size_t value_offset = 0;
                std::size_t label_offset = 0;
                std::size_t class_coordinate = 0;

                for (std::size_t i = index.rank; i > 0; --i) {
                    const auto axis = i - 1;
                    if (axis == index.axis && !scatter) {
                        continue;
                    }

                    const auto coordinate = remaining % index.dims[axis];
                    remaining /= index.dims[axis];
                    if (axis == index.axis) {
                        class_coordinate = coordinate;
                        continue;
                    }

                    value_offset += coordinate * index.value_strides[axis];
                    label_offset += coordinate * index.label_strides[axis];
                }

                const auto label = labels[label_offset];
                if (index.has_ignore_index && label == index.ignore_index) {
                    out[linear] = stored<T>(0);
                    continue;
                }
                if (label < 0 || static_cast<std::size_t>(label) >= index.dims[index.axis]) {
                    out[linear] = stored<T>(nanf(""));
                    continue;
                }

                if constexpr (scatter) {
                    if (static_cast<std::size_t>(label) == class_coordinate) {
                        out[linear] = input[value_offset];
                    } else {
                        out[linear] = stored<T>(0);
                    }
                } else {
                    const auto class_offset =
                        static_cast<std::size_t>(label) * index.value_strides[index.axis];
                    out[linear] = input[value_offset + class_offset];
                }
            }
        }

        template <typename T>
        __global__ void scatter_scalar_kernel(T* out, const T* scalar, std::size_t count, std::size_t index) {
            const auto step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
            for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 linear < count;
                 linear += step) {
                out[linear] = linear == index ? *scalar : stored<T>(0);
            }
        }

        template <bool scatter>
        void select(
            core::Tensor& out,
            const core::Tensor& input,
            const core::Tensor& indices,
            const core::Pick& attrs
        ) {
            if (out.numel() == 0) {
                return;
            }
            if (attrs.shape.rank() > 64) {
                ORMI_THROW(ShapeError, "CUDA indexed selection supports at most 64 dimensions.");
            }

            Index index{};
            index.rank = attrs.shape.rank();
            index.axis = static_cast<std::size_t>(attrs.axis);
            index.has_ignore_index = attrs.ignore_index.has_value();
            index.ignore_index = attrs.ignore_index.value_or(0);
            const auto value_strides = input.strides();
            const auto label_strides = indices.strides();

            for (std::size_t axis = 0; axis < index.rank; ++axis) {
                index.dims[axis] = static_cast<std::size_t>(attrs.shape[axis]);
                if (axis == index.axis) {
                    if constexpr (!scatter) {
                        index.value_strides[axis] = static_cast<std::size_t>(value_strides[axis]);
                    }
                    continue;
                }

                const auto label_axis = axis > index.axis ? axis - 1 : axis;
                const auto value_axis = scatter ? label_axis : axis;
                index.value_strides[axis] = static_cast<std::size_t>(value_strides[value_axis]);
                index.label_strides[axis] = static_cast<std::size_t>(label_strides[label_axis]);
            }

            auto& device = state(out.device());
            check(cudaSetDevice(device.index()), "cudaSetDevice");
            const auto blocks = elementwise::blocks(out.numel());
            const auto* labels = static_cast<const std::int32_t*>(indices.data());
            switch (out.dtype()) {
                case core::DType::float64:
                    select_kernel<double, scatter><<<blocks, 256, 0, device.stream()>>>(
                        static_cast<double*>(out.data()),
                        static_cast<const double*>(input.data()),
                        labels,
                        out.numel(),
                        index
                    );
                    break;
                case core::DType::float32:
                    select_kernel<float, scatter><<<blocks, 256, 0, device.stream()>>>(
                        static_cast<float*>(out.data()),
                        static_cast<const float*>(input.data()),
                        labels,
                        out.numel(),
                        index
                    );
                    break;
                case core::DType::float16:
                    select_kernel<__half, scatter><<<blocks, 256, 0, device.stream()>>>(
                        static_cast<__half*>(out.data()),
                        static_cast<const __half*>(input.data()),
                        labels,
                        out.numel(),
                        index
                    );
                    break;
                default:
                    ORMI_THROW(DTypeError, "Indexed selection requires floating-point input.");
            }
            check(cudaGetLastError(), "CUDA indexed selection");
        }
    }

    void scatter_scalar(core::Tensor& out, const core::Tensor& scalar, std::size_t index) {
        if (out.numel() == 0) {
            return;
        }

        auto& device = state(out.device());
        check(cudaSetDevice(device.index()), "cudaSetDevice");
        const auto blocks = elementwise::blocks(out.numel());
        switch (out.dtype()) {
            case core::DType::float64:
                scatter_scalar_kernel<double><<<blocks, 256, 0, device.stream()>>>(
                    static_cast<double*>(out.data()),
                    static_cast<const double*>(scalar.data()),
                    out.numel(),
                    index
                );
                break;
            case core::DType::float32:
                scatter_scalar_kernel<float><<<blocks, 256, 0, device.stream()>>>(
                    static_cast<float*>(out.data()),
                    static_cast<const float*>(scalar.data()),
                    out.numel(),
                    index
                );
                break;
            case core::DType::float16:
                scatter_scalar_kernel<__half><<<blocks, 256, 0, device.stream()>>>(
                    static_cast<__half*>(out.data()),
                    static_cast<const __half*>(scalar.data()),
                    out.numel(),
                    index
                );
                break;
            default:
                ORMI_THROW(DTypeError, "CUDA scalar scatter requires floating-point input.");
        }
        check(cudaGetLastError(), "CUDA scalar scatter");
    }

    void pick(
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor& indices,
        const core::Pick& attrs
    ) {
        select<false>(out, input, indices, attrs);
    }

    void unpick(
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor& indices,
        const core::Pick& attrs
    ) {
        select<true>(out, input, indices, attrs);
    }
}
