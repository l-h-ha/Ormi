#include <cublas_v2.h>
#include <sstream>
#include <unordered_map>
#include <mutex>

#include "jit.cuh"
#include "compiler.cuh"
#include "utils.cuh"
#include "utils/macros.h"
#include "core/shape.h"
#include "./kernels/reduce/sum.cuh"
#include "./kernels/blas/matmul.cuh"
#include "backend/cuda/config.cuh"

namespace ormi::cuda {
    static std::unordered_map<std::string, CUfunction> jit_cache;
    static std::mutex jit_mutex;

    void execute_jit(const std::shared_ptr<Node>& root) {
        CompilationResult ast = compile(root);

        CUdevice cuDevice;
        CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
        CUfunction kernel;

        {
            std::lock_guard<std::mutex> lock(jit_mutex);
            auto it = jit_cache.find(ast.kernel_code);
            if (it != jit_cache.end()) {
                kernel = it->second;
            } else {
                std::vector<char> ptx = compile_ptx(ast.kernel_code, cuDevice);
                CUmodule module;
                CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
                CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "fused_kernel"));
                jit_cache[ast.kernel_code] = kernel;
            }
        }

        if (!root->data) {
            root->data = empty(ast.out_shape, root->device, root->dtype);
            root->view.shape = ast.out_shape;
            root->view.strides = calc_contiguous_strides(ast.out_shape);
        }

        root->view.shape = ast.out_shape;
        root->view.strides = calc_contiguous_strides(ast.out_shape);

        void* out_ptr = root->data->ptr;

        std::vector<void*> leaf_ptrs;
        leaf_ptrs.reserve(ast.leaf_nodes.size());

        std::vector<void*> args;
        args.reserve(ast.leaf_nodes.size() + 1);

        args.push_back(&out_ptr);

        for (auto& node : ast.leaf_nodes) {
            leaf_ptrs.push_back(node->data->ptr);
            args.push_back(&leaf_ptrs.back());
        }

        const int threads = get_elementwise_threads();
        const int blocks = static_cast<int>(
            (static_cast<size_t>(ast.tensor_size) + threads - 1) / threads
        );

        CUDA_SAFE_CALL(cuLaunchKernel(
            kernel,
            blocks, 1, 1,
            threads, 1, 1,
            0,
            cuda::get_stream(),
            args.data(),
            0
        ));
    }
}