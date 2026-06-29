#include <memory>
#include <cublas_v2.h>
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <vector>

#include "core/node.h"
#include "core/allocator.h"
#include "sum.cuh"
#include "backend/cuda/utils.cuh"
#include "utils/macros.h"
#include "backend/graph_breakers.h"
#include "backend/cuda/config.cuh"

namespace ormi::cuda {
    static std::unordered_map<std::string, CUfunction> kernel_cache;
    static std::mutex cache_mutex;

    std::shared_ptr<Buffer> sum(
        const std::shared_ptr<Buffer>& in_buf,
        const TensorView& in_view,
        const std::vector<int>& axes,
        const TensorView& out_view,
        const bool keepdims,
        std::shared_ptr<Buffer> out_buffer
    ) {
        int in_elements = in_buf->bytes / dtype_size(in_buf->dtype);

        if (!out_buffer) {
            out_buffer = zeros(out_view.shape, in_buf->device, in_buf->dtype);
        }

        CUdevice cuDevice;
        CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

        if (in_buf->dtype != DType::Float32 && in_buf->dtype != DType::Float16) {
            ORMI_THROW(DTypeError, "CUDA SUM only supports Float32 and Float16.");
        }

        std::string type_str = (in_buf->dtype == DType::Float32) ? "float" : "__half";
        std::stringstream src;

        if (in_buf->dtype == DType::Float16) {
            src << "#include <cuda_fp16.h>\n";
        }

        bool is_full_reduce = axes.empty() || axes.size() == in_view.shape.size();

        if (is_full_reduce) {
            src << "extern \"C\" __global__ void sum_kernel(const " << type_str << "* IN, " << type_str << "* OUT) {\n";
            src << "    extern __shared__ float sdata[];\n";
            src << "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
            src << "    int tid = threadIdx.x;\n";

            src << "    sdata[tid] = (i < " << in_elements << ") ? (float)IN[i] : 0.0f;\n";
            src << "    __syncthreads();\n";

            src << "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n";
            src << "        if (tid < s) {\n";
            src << "            sdata[tid] += sdata[tid + s];\n";
            src << "        }\n";
            src << "        __syncthreads();\n";
            src << "    }\n";

            src << "    if (tid == 0) {\n";
            src << "        atomicAdd(&OUT[0], (" << type_str << ")sdata[0]);\n";
            src << "    }\n";
            src << "}\n";
        } else {
            src << "extern \"C\" __global__ void sum_kernel(const " << type_str << "* IN, " << type_str << "* OUT) {\n";
            src << "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
            src << "    if (i >= " << in_elements << ") return;\n";
            src << "    int remaining = i;\n";
            src << "    int out_flat = 0;\n";

            int out_d = 0;
            for (size_t d = 0; d < in_view.shape.size(); ++d) {
                src << "    {\n";
                src << "        int coord = remaining / " << in_view.strides[d] << ";\n";
                src << "        remaining = remaining % " << in_view.strides[d] << ";\n";

                bool is_reduced = false;
                for (int ax : axes) {
                    if (ax == d) { is_reduced = true; break; }
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
        }

        std::string kernel_code = src.str();
        CUfunction kernel;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = kernel_cache.find(kernel_code);

            if (it != kernel_cache.end()) {
                kernel = it->second;
            } else {
                std::vector<char> ptx = compile_ptx(kernel_code, cuDevice);
                CUmodule module;
                CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
                CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "sum_kernel"));

                kernel_cache[kernel_code] = kernel;
            }
        }

        void* in_ptr = in_buf->ptr;
        void* out_ptr = out_buffer->ptr;

        void* args[] = {
            &in_ptr,
            &out_ptr
        };

        const int threads = get_reduction_threads();

        const int blocks = static_cast<int>(
            (static_cast<size_t>(in_elements) + threads - 1) / threads
        );

        const unsigned int shared_mem_bytes =
            is_full_reduce ? static_cast<unsigned int>(threads * sizeof(float)) : 0;

        CUDA_SAFE_CALL(cuLaunchKernel(
            kernel,
            blocks, 1, 1,
            threads, 1, 1,
            shared_mem_bytes,
            cuda::get_stream(),
            args,
            0
        ));

        return out_buffer;
    }

    void sum_handler(const std::shared_ptr<Node>& root) {
        const auto& attrs = std::get<ReduceAttrs>(root->attrs);

        if (!root->data) {
            root->data = zeros(root->view.shape, root->device, root->dtype);
        } else {
            get_allocator(root->device)->memset(
                root->data->ptr,
                0,
                root->data->bytes
            );
        }

        cuda::sum(
            root->parents[0]->data,
            root->parents[0]->view,
            attrs.axes,
            root->view,
            attrs.keepdims,
            root->data
        );
    }

    static RegisterGraphBreaker reg_cuda_sum(DeviceEnum::CUDA, ops::OpEnum::SUM, sum_handler);
}