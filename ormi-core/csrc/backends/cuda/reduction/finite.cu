#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/allocator.h"
#include "backends/cuda/check.cuh"
#include "backends/cuda/elementwise/index.cuh"
#include "backends/cuda/reduction/finite.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        inline constexpr std::size_t values_per_thread = 8;

        struct Chunk {
            const void* data;
            std::size_t elements;
            std::size_t part;
            std::size_t parts;
            core::DType dtype;
        };

        struct Contiguous {
            __device__ std::size_t offset(std::size_t linear) const {
                return linear;
            }
        };

        struct Indexed {
            elementwise::Index index;

            __device__ std::size_t offset(std::size_t linear) const {
                return elementwise::storage_offset(linear, index);
            }
        };

        template <typename T>
        __device__ bool finite(T value) {
            if constexpr (std::is_same_v<T, __half>) {
                return ::isfinite(__half2float(value));
            } else {
                return ::isfinite(value);
            }
        }

        template <typename T, typename Layout>
        __global__ void finite_kernel(
            const T* input,
            std::size_t elements,
            Layout layout,
            int* invalid
        ) {
            std::size_t linear =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

            for (; linear < elements; linear += stride) {
                if (!finite(input[layout.offset(linear)])) {
                    atomicExch(invalid, 1);
                }
            }
        }

        template <typename T>
        __device__ void scan_chunk(const Chunk& chunk, int* invalid) {
            const auto* input = static_cast<const T*>(chunk.data);

            for (
                std::size_t linear =
                    chunk.part * blockDim.x + threadIdx.x;
                linear < chunk.elements;
                linear += chunk.parts * blockDim.x
            ) {
                if (!finite(input[linear])) {
                    atomicExch(invalid, 1);
                }
            }
        }

        __global__ void finite_chunks_kernel(
            const Chunk* chunks,
            std::size_t count,
            int* invalid
        ) {
            for (
                std::size_t index = blockIdx.x;
                index < count;
                index += gridDim.x
            ) {
                const auto chunk = chunks[index];

                switch (chunk.dtype) {
                    case core::DType::float64:
                        scan_chunk<double>(chunk, invalid);
                        break;
                    case core::DType::float32:
                        scan_chunk<float>(chunk, invalid);
                        break;
                    case core::DType::float16:
                        scan_chunk<__half>(chunk, invalid);
                        break;
                    case core::DType::int32:
                    case core::DType::boolean:
                        break;
                }
            }
        }

        std::size_t chunk_count(std::size_t elements) {
            const auto elements_per_chunk =
                elementwise::threads * values_per_thread;
            return std::min<std::size_t>(
                elementwise::max_blocks,
                elements / elements_per_chunk +
                    (elements % elements_per_chunk != 0)
            );
        }

        void append_chunks(std::vector<Chunk>& chunks, const core::Tensor& input) {
            if (input.numel() == 0) {
                return;
            }

            const auto parts = chunk_count(input.numel());
            const auto* data = input.data();

            for (std::size_t part = 0; part < parts; ++part) {
                chunks.push_back(Chunk{
                    .data = data,
                    .elements = input.numel(),
                    .part = part,
                    .parts = parts,
                    .dtype = input.dtype()
                });
            }
        }

        template <typename T>
        void launch_scan(
            const core::Tensor& input,
            int* invalid,
            cudaStream_t stream
        ) {
            if (input.numel() == 0) {
                return;
            }

            const auto* data = static_cast<const T*>(input.data());
            const auto blocks = elementwise::blocks(input.numel());

            if (input.is_contiguous()) {
                finite_kernel<T><<<blocks, elementwise::threads, 0, stream>>>(
                    data,
                    input.numel(),
                    Contiguous{},
                    invalid
                );
            } else {
                const auto index = elementwise::make_index(input);
                finite_kernel<T><<<blocks, elementwise::threads, 0, stream>>>(
                    data,
                    input.numel(),
                    Indexed{index},
                    invalid
                );
            }

            check(cudaGetLastError(), "CUDA all-finite kernel");
        }

        void launch_scan(
            const core::Tensor& input,
            int* invalid,
            cudaStream_t stream
        ) {
            switch (input.dtype()) {
                case core::DType::float64:
                    launch_scan<double>(input, invalid, stream);
                    return;
                case core::DType::float32:
                    launch_scan<float>(input, invalid, stream);
                    return;
                case core::DType::float16:
                    launch_scan<__half>(input, invalid, stream);
                    return;
                case core::DType::int32:
                case core::DType::boolean:
                    ORMI_THROW(
                        DTypeError,
                        "CUDA all-finite supports floating-point tensors only."
                    );
            }

            ORMI_THROW(DTypeError, "CUDA all-finite cannot handle an unknown dtype.");
        }
    }

    bool all_finite(std::span<const core::Tensor* const> inputs) {
        const auto device = inputs.front()->device();
        auto& cuda_state = state(device);
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

        std::vector<Chunk> chunks;
        std::vector<const core::Tensor*> indexed;
        chunks.reserve(inputs.size());
        indexed.reserve(inputs.size());

        for (const auto* input : inputs) {
            if (inputs.size() > 1 && input->is_contiguous()) {
                append_chunks(chunks, *input);
            } else {
                indexed.push_back(input);
            }
        }

        const auto chunks_bytes = chunks.size() * sizeof(Chunk);
        auto* memory = allocator(device);
        int* invalid = nullptr;
        Chunk* device_chunks = nullptr;
        int host_invalid = 0;

        try {
            invalid = static_cast<int*>(memory->allocate(sizeof(int)));

            if (!chunks.empty()) {
                device_chunks = static_cast<Chunk*>(memory->allocate(chunks_bytes));
            }

            check(
                cudaMemsetAsync(invalid, 0, sizeof(int), cuda_state.stream()),
                "cudaMemsetAsyncFiniteStatus"
            );

            if (!chunks.empty()) {
                check(
                    cudaMemcpyAsync(
                        device_chunks,
                        chunks.data(),
                        chunks_bytes,
                        cudaMemcpyHostToDevice,
                        cuda_state.stream()
                    ),
                    "cudaMemcpyAsyncFiniteChunks"
                );
                finite_chunks_kernel<<<
                    elementwise::blocks(chunks.size()),
                    elementwise::threads,
                    0,
                    cuda_state.stream()
                >>>(device_chunks, chunks.size(), invalid);
                check(cudaGetLastError(), "CUDA batched all-finite kernel");
            }

            for (const auto* input : indexed) {
                launch_scan(*input, invalid, cuda_state.stream());
            }

            check(
                cudaMemcpyAsync(
                    &host_invalid,
                    invalid,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    cuda_state.stream()
                ),
                "cudaMemcpyAsyncFiniteStatus"
            );
            check(
                cudaStreamSynchronize(cuda_state.stream()),
                "cudaStreamSynchronizeFiniteStatus"
            );
        } catch (...) {
            memory->deallocate(device_chunks, chunks_bytes);
            memory->deallocate(invalid, sizeof(int));
            throw;
        }

        memory->deallocate(device_chunks, chunks_bytes);
        memory->deallocate(invalid, sizeof(int));
        return host_invalid == 0;
    }
}
