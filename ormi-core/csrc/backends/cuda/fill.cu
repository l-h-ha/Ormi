#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_runtime_api.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/memory.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        constexpr unsigned int fill_threads = 256;
        constexpr std::size_t max_fill_blocks = 65535;

        template <typename T>
        __global__ void fill_kernel(T* out, T value, std::size_t elements) {
            std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; index < elements; index += stride) {
                out[index] = value;
            }
        }

        template <typename T>
        void fill_typed(
            void* dst,
            const void* value,
            std::size_t elements,
            cudaStream_t stream
        ) {
            T typed_value;
            std::memcpy(&typed_value, value, sizeof(T));

            const std::size_t required_blocks = elements / fill_threads + (elements % fill_threads != 0);
            const auto blocks = static_cast<unsigned int>(std::min(required_blocks, max_fill_blocks));

            fill_kernel<<<blocks, fill_threads, 0, stream>>>(
                static_cast<T*>(dst),
                typed_value,
                elements
            );
            check(cudaGetLastError(), "fill_kernel");
        }
    }

    void fill_scalar(
        core::Device device,
        void* dst,
        const void* value,
        std::size_t elements,
        core::DType dtype
    ) {
        if (elements == 0) {
            return;
        }

        if (device.is_cpu()) {
            ORMI_THROW(cudaError, "CUDA cannot fill a tensor on a CPU device.");
        }

        auto& cuda_state = state(device);
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

        switch (dtype) {
            case core::DType::float64:
                fill_typed<double>(dst, value, elements, cuda_state.stream());
                break;
            case core::DType::float32:
                fill_typed<float>(dst, value, elements, cuda_state.stream());
                break;
            case core::DType::float16:
                fill_typed<std::uint16_t>(dst, value, elements, cuda_state.stream());
                break;
            case core::DType::int32:
                fill_typed<std::int32_t>(dst, value, elements, cuda_state.stream());
                break;
            case core::DType::boolean:
                fill_typed<bool>(dst, value, elements, cuda_state.stream());
                break;
            default:
                ORMI_THROW(DTypeError, "CUDA cannot fill an unknown dtype.");
        }

    }
}
