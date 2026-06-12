#include <memory>
#include <cublas_v2.h>
#include <sstream>

#include "core/node.h"
#include "cuda_sum.cuh"
#include "backend/cuda/utils.cuh"
#include "utils/macros.h"
#include "backend/graph_breakers.h"

namespace ormi::cuda {
    std::shared_ptr<Buffer> sum(
        std::shared_ptr<Buffer> in_buf, 
        const TensorView& in_view,
        const std::vector<int>& axes, 
        const TensorView& out_view,
        const bool keepdims
    ) {
        auto out_buffer = zeros(out_view.shape, DeviceEnum::CUDA);
        int in_elements = in_buf->bytes / sizeof(float);
    
        // Zeroing output buffer
        CUDA_SAFE_CALL(cuInit(0));
        CUdevice cuDevice; CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
        CUcontext cuContext; CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
        CUDA_SAFE_CALL(cuCtxSetCurrent(cuContext));
    
        // NVRTC kernel
        std::stringstream src;
        src << "extern \"C\" __global__ void sum_kernel(const float* IN, float* OUT) {\n";
        src << "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
        src << "    if (i >= " << in_elements << ") return;\n";
        src << "    int remaining = i;\n";
        src << "    int out_flat = 0;\n";
    
        int out_d = 0;

        // Dimension math using explicit view strides
        for (size_t d = 0; d < in_view.shape.size(); ++d) {
            src << "    {\n";
            src << "        int coord = remaining / " << in_view.strides[d] << ";\n";
            src << "        remaining = remaining % " << in_view.strides[d] << ";\n";
            
            bool is_reduced = false;
            if (axes.empty()) {
                is_reduced = true;
            } else {
                for (int ax : axes) {
                    if (ax == d) {
                        is_reduced = true;
                        break;
                    }
                }
            }
            
            if (!is_reduced) {
                src << "        out_flat += coord * " << out_view.strides[out_d] << ";\n";
                out_d++;
            } else if (keepdims) {
                out_d++;
            }
            src << "    }\n";
        }
        
        src << "    atomicAdd(&OUT[out_flat], IN[i]);\n"; 
        src << "}\n";
    
        std::vector<char> ptx = compile_ptx(src.str(), cuDevice);
        CUmodule module; CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
        CUfunction kernel; CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "sum_kernel"));
    
        std::vector<void*> args;
        args.push_back(&(in_buf->ptr));
        args.push_back(&(out_buffer->ptr));
    
        int threads = 256;
        int blocks = (in_elements + threads - 1) / threads;
        CUDA_SAFE_CALL(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, 0, args.data(), 0));
    
        cuModuleUnload(module);
        cuDevicePrimaryCtxRelease(cuDevice);
    
        return out_buffer;
    }

    void sum_handler(std::shared_ptr<Node> root) {
        const auto& attrs = std::get<ReduceAttrs>(root->attrs);
        root->data = cuda::sum(
            root->parents[0]->data, 
            root->parents[0]->view, 
            attrs.axes, 
            root->view, 
            attrs.keepdims
        );
    }

    static RegisterGraphBreaker reg_cuda_sum(DeviceEnum::CUDA, ops::OpEnum::SUM, sum_handler);
}