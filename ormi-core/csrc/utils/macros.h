#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvrtc.h>
#include <string>
#include "core/error.h"

namespace ormi {
    // cuBLAS string decoder
    inline const char* cublasGetErrorString(cublasStatus_t status) {
        switch(status) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
            default: return "UNKNOWN_CUBLAS_ERROR";
        }
    }
}

#define NVRTC_SAFE_CALL(x) do { \
    nvrtcResult _ormi_result = x; \
    if (_ormi_result != NVRTC_SUCCESS) { \
        ORMI_THROW(ormi::DeviceError, "NVRTC Error: " + std::string(nvrtcGetErrorString(_ormi_result))); \
    } \
} while(false)

#define CUDA_SAFE_CALL(x) do { \
    CUresult _ormi_result = x; \
    if (_ormi_result != CUDA_SUCCESS) { \
        const char* _ormi_msg = nullptr; \
        cuGetErrorString(_ormi_result, &_ormi_msg); \
        ORMI_THROW(ormi::DeviceError, "CUDA Driver Error: " + std::string(_ormi_msg ? _ormi_msg : "Unknown")); \
    } \
} while(false)

#define CUDART_SAFE_CALL(x) do { \
    cudaError_t _ormi_result = x; \
    if (_ormi_result != cudaSuccess) { \
        ORMI_THROW(ormi::DeviceError, "CUDA Runtime Error: " + std::string(cudaGetErrorString(_ormi_result))); \
    } \
} while(false)

#define CUBLAS_SAFE_CALL(x) do { \
    cublasStatus_t _ormi_result = x; \
    if (_ormi_result != CUBLAS_STATUS_SUCCESS) { \
        ORMI_THROW(ormi::DeviceError, "cuBLAS Error: " + std::string(ormi::cublasGetErrorString(_ormi_result))); \
    } \
} while(false)
