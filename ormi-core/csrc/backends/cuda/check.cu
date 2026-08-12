#include <string>
#include <string_view>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <ormi/core/error.h>

#include "check.cuh"

namespace ormi::backends::cuda {
    namespace {
        const char* cublas_status_name(cublasStatus_t status) {
            switch (status) {
                case CUBLAS_STATUS_SUCCESS:
                    return "CUBLAS_STATUS_SUCCESS";
                case CUBLAS_STATUS_NOT_INITIALIZED:
                    return "CUBLAS_STATUS_NOT_INITIALIZED";
                case CUBLAS_STATUS_ALLOC_FAILED:
                    return "CUBLAS_STATUS_ALLOC_FAILED";
                case CUBLAS_STATUS_INVALID_VALUE:
                    return "CUBLAS_STATUS_INVALID_VALUE";
                case CUBLAS_STATUS_ARCH_MISMATCH:
                    return "CUBLAS_STATUS_ARCH_MISMATCH";
                case CUBLAS_STATUS_MAPPING_ERROR:
                    return "CUBLAS_STATUS_MAPPING_ERROR";
                case CUBLAS_STATUS_EXECUTION_FAILED:
                    return "CUBLAS_STATUS_EXECUTION_FAILED";
                case CUBLAS_STATUS_INTERNAL_ERROR:
                    return "CUBLAS_STATUS_INTERNAL_ERROR";
                case CUBLAS_STATUS_NOT_SUPPORTED:
                    return "CUBLAS_STATUS_NOT_SUPPORTED";
                case CUBLAS_STATUS_LICENSE_ERROR:
                    return "CUBLAS_STATUS_LICENSE_ERROR";
            }

            return "unknown cuBLAS error";
        }
    }

    void check_driver(CUresult status, std::string_view operation) {
        if (status == CUDA_SUCCESS) {
            return;
        }

        const char* error_name = nullptr;
        const char* error_description = nullptr;
        (void)cuGetErrorName(status, &error_name);
        (void)cuGetErrorString(status, &error_description);

        ORMI_THROW(
            BackendError,
            std::string(operation) + " failed with " +
                (error_name ? error_name : "unknown CUDA driver error") + ": " +
                (error_description ? error_description : "no description available")
        );
    }

    void check(cudaError_t status, std::string_view operation) {
        if (status == cudaSuccess) {
            return;
        }

        const char* error_name = cudaGetErrorName(status);
        const char* error_description = cudaGetErrorString(status);

        ORMI_THROW(
            BackendError,
            std::string(operation) + " failed with " +
                (error_name ? error_name : "unknown CUDA error") + ": " +
                (error_description ? error_description : "no description available")
        );
    }

    void check(cublasStatus_t status, std::string_view operation) {
        if (status == CUBLAS_STATUS_SUCCESS) {
            return;
        }

        ORMI_THROW(
            BackendError,
            std::string(operation) + " failed with " + cublas_status_name(status) + "."
        );
    }
}
