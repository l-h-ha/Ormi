#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/allocator.h"
#include "backends/cuda/check.cuh"
#include "backends/cuda/reduction/index.cuh"
#include "backends/cuda/reduction/sum.h"
#include "backends/cuda/state.h"

namespace ormi::backends::cuda {
    namespace {
        inline constexpr unsigned int max_threads = 256;
        inline constexpr std::size_t max_blocks = 65535;
        inline constexpr std::size_t values_per_thread = 8;

        struct Contiguous {
            __device__ std::size_t input_base(
                std::size_t output,
                std::size_t reduced_elements
            ) const {
                return output * reduced_elements;
            }

            __device__ std::size_t input_offset(std::size_t reduced) const {
                return reduced;
            }
        };

        struct Indexed {
            reduction::Index index;

            __device__ std::size_t input_base(
                std::size_t output,
                std::size_t
            ) const {
                return reduction::input_base_offset(output, index);
            }

            __device__ std::size_t input_offset(std::size_t reduced) const {
                return reduction::reduced_offset(reduced, index);
            }
        };

        template <typename Storage, typename Accum>
        __device__ Accum load_value(const Storage* input, std::size_t offset) {
            if constexpr (std::is_same_v<Storage, __half>) {
                return __half2float(input[offset]);
            } else {
                return static_cast<Accum>(input[offset]);
            }
        }

        template <typename Storage, typename Accum>
        __device__ void store_value(Storage* out, std::size_t offset, Accum value) {
            if constexpr (std::is_same_v<Storage, __half>) {
                out[offset] = __float2half_rn(value);
            } else {
                out[offset] = static_cast<Storage>(value);
            }
        }

        template <typename Accum>
        __device__ Accum block_sum(Accum value, Accum* partial) {
            partial[threadIdx.x] = value;
            __syncthreads();

            for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {
                if (threadIdx.x < width) {
                    partial[threadIdx.x] += partial[threadIdx.x + width];
                }
                __syncthreads();
            }

            return partial[0];
        }

        template <typename Storage, typename Accum, typename Layout>
        __global__ void sum_kernel(
            Storage* out,
            const Storage* input,
            std::size_t outputs,
            std::size_t reduced_elements,
            Layout layout
        ) {
            extern __shared__ unsigned char shared_bytes[];
            auto* partial = reinterpret_cast<Accum*>(shared_bytes);

            for (
                std::size_t output = blockIdx.x;
                output < outputs;
                output += gridDim.x
            ) {
                const auto base = layout.input_base(output, reduced_elements);
                Accum sum = Accum{};

                for (
                    std::size_t reduced = threadIdx.x;
                    reduced < reduced_elements;
                    reduced += blockDim.x
                ) {
                    const auto offset = base + layout.input_offset(reduced);
                    sum += load_value<Storage, Accum>(input, offset);
                }

                const auto total = block_sum(sum, partial);

                if (threadIdx.x == 0) {
                    store_value<Storage, Accum>(out, output, total);
                }
            }
        }

        template <typename Storage, typename Accum>
        __global__ void partial_sum_kernel(
            Accum* partials,
            const Storage* input,
            std::size_t reduced_elements,
            std::size_t parts
        ) {
            extern __shared__ unsigned char shared_bytes[];
            auto* partial = reinterpret_cast<Accum*>(shared_bytes);
            const auto output = static_cast<std::size_t>(blockIdx.x) / parts;
            const auto part = static_cast<std::size_t>(blockIdx.x) % parts;
            const auto base = output * reduced_elements;
            Accum sum = Accum{};

            for (
                std::size_t reduced = part * blockDim.x + threadIdx.x;
                reduced < reduced_elements;
                reduced += parts * blockDim.x
            ) {
                sum += load_value<Storage, Accum>(input, base + reduced);
            }

            const auto total = block_sum(sum, partial);

            if (threadIdx.x == 0) {
                partials[blockIdx.x] = total;
            }
        }

        template <typename Storage, typename Accum>
        __global__ void finish_sum_kernel(
            Storage* out,
            const Accum* partials,
            std::size_t outputs,
            std::size_t parts
        ) {
            extern __shared__ unsigned char shared_bytes[];
            auto* partial = reinterpret_cast<Accum*>(shared_bytes);

            for (
                std::size_t output = blockIdx.x;
                output < outputs;
                output += gridDim.x
            ) {
                Accum sum = Accum{};

                for (std::size_t part = threadIdx.x; part < parts; part += blockDim.x) {
                    sum += partials[output * parts + part];
                }

                const auto total = block_sum(sum, partial);

                if (threadIdx.x == 0) {
                    store_value<Storage, Accum>(out, output, total);
                }
            }
        }

        bool is_contiguous_suffix(
            const core::Tensor& input,
            std::span<const std::int64_t> axes
        ) {
            if (!input.is_contiguous()) {
                return false;
            }

            const auto first = input.rank() - axes.size();

            for (std::size_t i = 0; i < axes.size(); ++i) {
                if (axes[i] != static_cast<std::int64_t>(first + i)) {
                    return false;
                }
            }

            return true;
        }

        std::size_t reduced_elements(
            const core::Tensor& input,
            std::span<const std::int64_t> axes
        ) {
            std::size_t elements = 1;

            for (const auto axis : axes) {
                elements *= static_cast<std::size_t>(
                    input.shape()[static_cast<std::size_t>(axis)]
                );
            }

            return elements;
        }

        unsigned int thread_count(std::size_t elements) {
            unsigned int threads = 1;

            while (threads < elements && threads < max_threads) {
                threads *= 2;
            }

            return threads;
        }

        std::size_t partial_count(std::size_t outputs, std::size_t elements) {
            const auto elements_per_block = max_threads * values_per_thread;
            const auto required = std::max<std::size_t>(
                1,
                elements / elements_per_block + (elements % elements_per_block != 0)
            );
            const auto available = outputs > max_blocks
                ? std::size_t{1}
                : std::max<std::size_t>(1, max_blocks / outputs);
            return std::min(required, available);
        }

        template <typename Storage, typename Accum>
        void launch_hierarchical_sum(
            core::Tensor& out,
            const core::Tensor& input,
            std::size_t reduction_size,
            std::size_t parts,
            cudaStream_t stream
        ) {
            const auto outputs = out.numel();
            const auto partials_count = outputs * parts;
            const auto partials_bytes = partials_count * sizeof(Accum);
            auto* memory = allocator(out.device());
            auto* partials = static_cast<Accum*>(memory->allocate(partials_bytes));

            try {
                partial_sum_kernel<Storage, Accum><<<
                    static_cast<unsigned int>(partials_count),
                    max_threads,
                    max_threads * sizeof(Accum),
                    stream
                >>>(
                    partials,
                    static_cast<const Storage*>(input.data()),
                    reduction_size,
                    parts
                );
                check(cudaGetLastError(), "CUDA partial sum kernel");

                const auto final_threads = thread_count(parts);
                finish_sum_kernel<Storage, Accum><<<
                    static_cast<unsigned int>(outputs),
                    final_threads,
                    final_threads * sizeof(Accum),
                    stream
                >>>(
                    static_cast<Storage*>(out.data()),
                    partials,
                    outputs,
                    parts
                );
                check(cudaGetLastError(), "CUDA final sum kernel");
            } catch (...) {
                memory->deallocate(partials, partials_bytes);
                throw;
            }

            memory->deallocate(partials, partials_bytes);
        }

        template <typename Storage, typename Accum>
        void launch_sum(
            core::Tensor& out,
            const core::Tensor& input,
            std::span<const std::int64_t> axes,
            cudaStream_t stream
        ) {
            const auto outputs = out.numel();
            const auto reduction_size = reduced_elements(input, axes);
            const auto threads = thread_count(reduction_size);
            const auto blocks = static_cast<unsigned int>(
                std::min(outputs, max_blocks)
            );
            const auto shared_bytes = threads * sizeof(Accum);
            auto* out_data = static_cast<Storage*>(out.data());
            const auto* input_data = static_cast<const Storage*>(input.data());

            if (is_contiguous_suffix(input, axes)) {
                const auto parts = partial_count(outputs, reduction_size);

                if (parts > 1) {
                    launch_hierarchical_sum<Storage, Accum>(
                        out,
                        input,
                        reduction_size,
                        parts,
                        stream
                    );
                    return;
                }

                sum_kernel<Storage, Accum><<<
                    blocks,
                    threads,
                    shared_bytes,
                    stream
                >>>(
                    out_data,
                    input_data,
                    outputs,
                    reduction_size,
                    Contiguous{}
                );
            } else {
                const auto index = reduction::make_index(input, axes);
                sum_kernel<Storage, Accum><<<
                    blocks,
                    threads,
                    shared_bytes,
                    stream
                >>>(
                    out_data,
                    input_data,
                    outputs,
                    reduction_size,
                    Indexed{index}
                );
            }

            check(cudaGetLastError(), "CUDA sum kernel");
        }
    }

    void reduce_sum(
        core::Tensor& out,
        const core::Tensor& input,
        std::span<const std::int64_t> axes
    ) {
        if (out.numel() == 0) {
            return;
        }

        auto& cuda_state = state(out.device());
        check(cudaSetDevice(cuda_state.index()), "cudaSetDevice");

        switch (out.dtype()) {
            case core::DType::float64:
                launch_sum<double, double>(out, input, axes, cuda_state.stream());
                return;
            case core::DType::float32:
                launch_sum<float, float>(out, input, axes, cuda_state.stream());
                return;
            case core::DType::float16:
                launch_sum<__half, float>(out, input, axes, cuda_state.stream());
                return;
            case core::DType::int32:
                launch_sum<std::int32_t, std::int32_t>(
                    out,
                    input,
                    axes,
                    cuda_state.stream()
                );
                return;
            case core::DType::boolean:
                ORMI_THROW(DTypeError, "CUDA sum does not support Bool.");
        }

        ORMI_THROW(DTypeError, "CUDA sum cannot handle an unknown dtype.");
    }
}
