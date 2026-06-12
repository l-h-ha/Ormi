#include <vector>
#include "utils.cuh"
#include "utils/macros.h"

namespace ormi::cuda {
    std::vector<char> compile_ptx(const std::string& kernel_code, CUdevice cuDevice) {
        int cc_major = 0, cc_minor = 0;
        CUDA_SAFE_CALL(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
        CUDA_SAFE_CALL(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
        
        std::string arch = "--gpu-architecture=compute_" + std::to_string(cc_major) + std::to_string(cc_minor);
        const char* opts[] = {arch.c_str()};
        
        nvrtcProgram prog;
        NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        
        if (nvrtcCompileProgram(prog, 1, opts) != NVRTC_SUCCESS) {
            size_t logSize; nvrtcGetProgramLogSize(prog, &logSize);
            std::vector<char> log(logSize); nvrtcGetProgramLog(prog, log.data());
            throw std::runtime_error("NVRTC Compilation failed: \n" + std::string(log.data()));
        }
    
        size_t ptxSize; NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        std::vector<char> ptx(ptxSize); NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
        return ptx;
    }
}