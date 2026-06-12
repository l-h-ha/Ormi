#include <cublas_v2.h>
#include <sstream> 

#include "jit.cuh"
#include "compiler.cuh"
#include "utils.cuh"
#include "utils/macros.h"
#include "core/shape.h"
#include "./kernels/reduce/cuda_sum.cuh"
#include "./kernels/blas/cuda_matmul.cuh"

namespace ormi::cuda {
    void execute_jit(std::shared_ptr<Node> root) {
        CompilationResult ast = compile(root);
        
        CUDA_SAFE_CALL(cuInit(0));
        CUdevice cuDevice; CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
        CUcontext cuContext; CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
        CUDA_SAFE_CALL(cuCtxSetCurrent(cuContext));

        std::vector<char> ptx = compile_ptx(ast.kernel_code, cuDevice);
        
        CUmodule module; CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
        CUfunction kernel; CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "fused_kernel"));

        root->data = empty(ast.out_shape, DeviceEnum::CUDA);
        root->view.shape = ast.out_shape;
        root->view.strides = calc_contiguous_strides(ast.out_shape);
        
        // Pack all leaf pointers to send to the GPU
        std::vector<void*> args;
        args.push_back(&(root->data->ptr));
        for (auto& node : ast.leaf_nodes) {
            args.push_back(&(node->data->ptr));
        }

        // Run kernel
        int threads = 256;
        int blocks = (ast.tensor_size + threads - 1) / threads;
        CUDA_SAFE_CALL(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, 0, args.data(), 0));

        cuModuleUnload(module);
        cuDevicePrimaryCtxRelease(cuDevice);
    }
}