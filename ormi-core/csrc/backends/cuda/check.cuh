#pragma once
#include <string_view>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace ormi::backends::cuda {
    void check(cudaError_t status, std::string_view operation);
    void check(cublasStatus_t status, std::string_view operation);
}
