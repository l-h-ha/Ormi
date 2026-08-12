#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include <ormi/core/attrs.h>
#include <ormi/core/tensor.h>

namespace ormi::backends::cuda::l2 {
    bool staged(std::size_t count) noexcept;

    void forward(
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor* initial_derivative,
        const core::Caputo& attrs,
        const double* weights,
        void* workspace,
        cudaStream_t stream
    );
}
