#pragma once
#include <string>
#include <vector>
#include <cuda.h>
#include <nvrtc.h>

namespace ormi::cuda {
    std::vector<char> compile_ptx(const std::string& kernel_code, CUdevice cuDevice);
}