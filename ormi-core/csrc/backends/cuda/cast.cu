#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/allocator.h"
#include "backends/cuda/cast.h"
#include "backends/cuda/check.cuh"
#include "backends/cuda/elementwise/index.cuh"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        template <typename T>
        inline constexpr bool is_half = std::is_same_v<T, __half>;

        template <typename To, typename From>
        inline constexpr bool checked_int_cast =
            std::is_same_v<To, std::int32_t> &&
            (std::is_floating_point_v<From> || is_half<From>);

        template <typename From>
        __device__ double floating_value(From value) {
            if constexpr (is_half<From>) {
                return static_cast<double>(__half2float(value));
            } else {
                return static_cast<double>(value);
            }
        }

        template <typename To, typename From>
        __device__ To convert(From value, int* invalid) {
            if constexpr (checked_int_cast<To, From>) {
                constexpr double int32_min = -2147483648.0;
                constexpr double int32_max = 2147483647.0;
                const auto number = floating_value(value);

                if (
                    !::isfinite(number) ||
                    number < int32_min ||
                    number > int32_max
                ) {
                    atomicExch(invalid, 1);
                    return 0;
                }

                return static_cast<std::int32_t>(number);
            } else if constexpr (is_half<To>) {
                if constexpr (is_half<From>) {
                    return value;
                } else {
                    return __float2half_rn(static_cast<float>(value));
                }
            } else if constexpr (is_half<From>) {
                const auto number = __half2float(value);

                if constexpr (std::is_same_v<To, bool>) {
                    return number != 0.0f;
                } else {
                    return static_cast<To>(number);
                }
            } else {
                return static_cast<To>(value);
            }
        }

        template <typename To, typename From>
        __global__ void cast_contiguous_kernel(
            To* out,
            const From* input,
            std::size_t elements,
            int* invalid
        ) {
            std::size_t linear =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                out[linear] = convert<To>(input[linear], invalid);
            }
        }

        template <typename To, typename From>
        __global__ void cast_strided_kernel(
            To* out,
            const From* input,
            std::size_t elements,
            elementwise::Index index,
            int* invalid
        ) {
            std::size_t linear =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                const auto input_offset = elementwise::storage_offset(linear, index);
                out[linear] = convert<To>(input[input_offset], invalid);
            }
        }

        template <typename To, typename From>
        void submit_cast(
            core::Tensor& out,
            const core::Tensor& input,
            cudaStream_t stream,
            int* invalid
        ) {
            const auto elements = out.numel();
            auto* out_data = static_cast<To*>(out.data());
            const auto* input_data = static_cast<const From*>(input.data());
            const auto grid = elementwise::blocks(elements);

            if (input.is_contiguous()) {
                cast_contiguous_kernel<<<grid, elementwise::threads, 0, stream>>>(
                    out_data,
                    input_data,
                    elements,
                    invalid
                );
            } else {
                const auto index = elementwise::make_index(input);
                cast_strided_kernel<<<grid, elementwise::threads, 0, stream>>>(
                    out_data,
                    input_data,
                    elements,
                    index,
                    invalid
                );
            }

            check(cudaGetLastError(), "CUDA cast kernel");
        }

        template <typename To, typename From>
        void launch_cast(
            core::Tensor& out,
            const core::Tensor& input,
            State& cuda_state
        ) {
            if constexpr (!checked_int_cast<To, From>) {
                submit_cast<To, From>(out, input, cuda_state.stream(), nullptr);
                return;
            }

            auto* memory = allocator(out.device());
            auto* invalid = static_cast<int*>(memory->allocate(sizeof(int)));
            int host_invalid = 0;

            try {
                check(
                    cudaMemsetAsync(invalid, 0, sizeof(int), cuda_state.stream()),
                    "cudaMemsetAsyncCastStatus"
                );
                submit_cast<To, From>(out, input, cuda_state.stream(), invalid);
                check(
                    cudaMemcpyAsync(
                        &host_invalid,
                        invalid,
                        sizeof(int),
                        cudaMemcpyDeviceToHost,
                        cuda_state.stream()
                    ),
                    "cudaMemcpyAsyncCastStatus"
                );
                check(
                    cudaStreamSynchronize(cuda_state.stream()),
                    "cudaStreamSynchronizeCastStatus"
                );
            } catch (...) {
                memory->deallocate(invalid, sizeof(int));
                throw;
            }

            memory->deallocate(invalid, sizeof(int));

            if (host_invalid != 0) {
                ORMI_THROW(
                    DTypeError,
                    "CUDA cast to Int32 requires finite values within the Int32 range."
                );
            }
        }

        template <typename From>
        void cast_from(core::Tensor& out, const core::Tensor& input, State& cuda_state) {
            switch (out.dtype()) {
                case core::DType::float64:
                    launch_cast<double, From>(out, input, cuda_state);
                    return;
                case core::DType::float32:
                    launch_cast<float, From>(out, input, cuda_state);
                    return;
                case core::DType::float16:
                    launch_cast<__half, From>(out, input, cuda_state);
                    return;
                case core::DType::int32:
                    launch_cast<std::int32_t, From>(out, input, cuda_state);
                    return;
                case core::DType::boolean:
                    launch_cast<bool, From>(out, input, cuda_state);
                    return;
            }

            ORMI_THROW(DTypeError, "CUDA cast cannot handle an unknown output dtype.");
        }
    }

    void cast(core::Tensor& out, const core::Tensor& input) {
        if (out.numel() == 0) {
            return;
        }

        auto& cuda_state = state(out.device());
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

        switch (input.dtype()) {
            case core::DType::float64:
                cast_from<double>(out, input, cuda_state);
                return;
            case core::DType::float32:
                cast_from<float>(out, input, cuda_state);
                return;
            case core::DType::float16:
                cast_from<__half>(out, input, cuda_state);
                return;
            case core::DType::int32:
                cast_from<std::int32_t>(out, input, cuda_state);
                return;
            case core::DType::boolean:
                cast_from<bool>(out, input, cuda_state);
                return;
        }

        ORMI_THROW(DTypeError, "CUDA cast cannot handle an unknown input dtype.");
    }
}
