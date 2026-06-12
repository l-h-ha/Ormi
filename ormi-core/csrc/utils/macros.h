#pragma once
#include <cuda.h>
#include <nvrtc.h>
#include <stdexcept>
#include <string>

namespace ormi {
    #define NVRTC_SAFE_CALL(x) do { \
        nvrtcResult result = x; \
        if (result != NVRTC_SUCCESS) throw std::runtime_error("NVRTC Error: " + std::string(nvrtcGetErrorString(result))); \
    } while(0)
    
    #define CUDA_SAFE_CALL(x) do { \
        CUresult result = x; \
        if (result != CUDA_SUCCESS) { \
            const char *msg; cuGetErrorString(result, &msg); \
            throw std::runtime_error("CUDA Error: " + std::string(msg)); \
        } \
    } while(0)
}