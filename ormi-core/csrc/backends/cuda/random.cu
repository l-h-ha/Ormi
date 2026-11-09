#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ormi/core/attrs.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/check.cuh"
#include "backends/cuda/random.h"
#include "backends/cuda/state.h"

/* NVIDIA cuRAND device-code disclaimer and U.S. Government End Users notice:
 * Copyright 2010-2014 NVIDIA Corporation. All rights reserved.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE. THEY ARE
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users. These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
 * only as a commercial end item. Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 */

namespace ormi::backends::cuda {
    namespace {
        template <typename T>
        __global__ void mask_kernel(
            T* output,
            std::size_t count,
            double probability,
            unsigned long long seed,
            const std::uint64_t* counter
        ) {
            const auto execution = *counter;
            const auto step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
            for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
                 i += step) {
                curandStatePhilox4_32_10_t generator;
                curand_init(seed, i, execution * 4ULL, &generator);
                // Convert (0, 1] to [0, 1), matching the probability comparison on CPU.
                const auto uniform = 1.0 - curand_uniform_double(&generator);
                output[i] = uniform < probability ? T(0) : static_cast<T>(1.0 / (1.0 - probability));
            }
        }

        __global__ void advance_counter(std::uint64_t* counter) {
            ++*counter;
        }
    }

    void bernoulli(core::Tensor& out, const core::Bernoulli& attrs) {
        auto& device = state(out.device());
        check(cudaSetDevice(device.index()), "cudaSetDevice");
        auto* counter = static_cast<std::uint64_t*>(attrs.counter->data());
        if (out.numel() != 0) {
            constexpr unsigned threads = 256;
            const auto blocks =
                static_cast<unsigned>(std::min<std::size_t>((out.numel() + threads - 1) / threads, 65535));
            if (out.dtype() == core::DType::float32) {
                mask_kernel<<<blocks, threads, 0, device.stream()>>>(
                    static_cast<float*>(out.data()),
                    out.numel(),
                    attrs.probability,
                    attrs.seed,
                    counter
                );
            } else if (out.dtype() == core::DType::float64) {
                mask_kernel<<<blocks, threads, 0, device.stream()>>>(
                    static_cast<double*>(out.data()),
                    out.numel(),
                    attrs.probability,
                    attrs.seed,
                    counter
                );
            } else {
                ORMI_THROW(DTypeError, "Bernoulli masks require Float32 or Float64 storage.");
            }
            check(cudaGetLastError(), "Dropout mask kernel");
        }
        // Separate kernels keep every thread on the same counter value, including graph replay.
        advance_counter<<<1, 1, 0, device.stream()>>>(counter);
        check(cudaGetLastError(), "Dropout counter kernel");
    }
}
