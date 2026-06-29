#pragma once
#include <string>
#include <vector>
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include "core/buffer.h"
#include "core/shape.h"

namespace ormi::cuda {
    cudaStream_t get_stream();
    cublasHandle_t get_cublas_handle();
    bool supports_tf32();
    std::vector<char> compile_ptx(const std::string& kernel_code, CUdevice cuDevice);
    std::shared_ptr<Buffer> make_contiguous(const std::shared_ptr<Buffer>& in_buf, const TensorView& view);
}