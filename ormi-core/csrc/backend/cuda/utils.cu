#include <vector>
#include <sstream>
#include <mutex>
#include <unordered_map>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.cuh"
#include "utils.h"
#include "utils/macros.h"
#include "core/buffer.h"
#include "core/shape.h"
#include "core/dtype.h"
#include "backend/cuda/config.cuh"

namespace ormi::cuda {
    cudaStream_t get_stream() {
        static cudaStream_t stream = nullptr;
        if (stream == nullptr) {
            CUDART_SAFE_CALL(cudaStreamCreate(&stream));
        }
        return stream;
    }

    cublasHandle_t get_cublas_handle() {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            CUBLAS_SAFE_CALL(cublasCreate(&handle));
            CUBLAS_SAFE_CALL(cublasSetStream(handle, get_stream()));
        }
        return handle;
    }

    bool supports_tf32() {
        int device_id = 0;
        CUDART_SAFE_CALL(cudaGetDevice(&device_id));

        cudaDeviceProp prop{};
        CUDART_SAFE_CALL(cudaGetDeviceProperties(&prop, device_id));

        return prop.major >= 8;
    }

    std::vector<char> compile_ptx(const std::string& kernel_code, CUdevice cuDevice) {
        int cc_major = 0, cc_minor = 0;
        CUDA_SAFE_CALL(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
        CUDA_SAFE_CALL(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

        std::string arch = "--gpu-architecture=compute_" + std::to_string(cc_major) + std::to_string(cc_minor);
        static const std::string include_opt = []() {
            if (const char* env_cuda_path = std::getenv("CUDA_PATH")) {
                return "-I" + std::string(env_cuda_path) + "/include";
            } else {
                return std::string("-I/usr/local/cuda/include");
            }
        }();

        const char* opts[] = {
            arch.c_str(),
            include_opt.c_str()
        };

        nvrtcProgram prog;
        NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_code.c_str(), "kernel.cu", 0, nullptr, nullptr));

        if (nvrtcCompileProgram(prog, 2, opts) != NVRTC_SUCCESS) {
            size_t logSize;
            nvrtcGetProgramLogSize(prog, &logSize);
            std::vector<char> log(logSize);
            nvrtcGetProgramLog(prog, log.data());
            nvrtcDestroyProgram(&prog);
            ORMI_THROW(ormi::DeviceError, "NVRTC Compilation failed:\n" + std::string(log.data()));
        }

        size_t ptxSize;
        NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
        std::vector<char> ptx(ptxSize);
        NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

        return ptx;
    }

    static std::unordered_map<std::string, CUfunction> contig_cache;
    static std::mutex contig_mutex;

    std::shared_ptr<Buffer> make_contiguous(const std::shared_ptr<Buffer>& in_buf, const TensorView& view) {
        if (view.is_contiguous()) return in_buf;

        auto out_buffer = empty(view.shape, in_buf->device, in_buf->dtype);
        const int elements = static_cast<int>(
            out_buffer->bytes / dtype_size(in_buf->dtype)
        );

        CUdevice cuDevice;
        CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
        CUfunction kernel;

        std::stringstream src;
        std::string type_str = (in_buf->dtype == DType::Float32) ? "float" : "__half";

        if (in_buf->dtype == DType::Float16) {
            src << "#include <cuda_fp16.h>\n";
        }

        src << "extern \"C\" __global__ void contig_kernel(const " << type_str << "* IN, " << type_str << "* OUT) {\n";
        src << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        src << "    if (idx >= " << elements << ") return;\n";

        int ndim = view.shape.size();
        src << "    int offset = 0;\n";
        src << "    int temp = idx;\n";
        src << "    int coord;\n";

        for (int i = ndim - 1; i >= 0; --i) {
            src << "    coord = temp % " << view.shape[i] << ";\n";
            src << "    temp /= " << view.shape[i] << ";\n";
            src << "    offset += coord * " << view.strides[i] << ";\n";
        }
        src << "    OUT[idx] = IN[offset];\n";
        src << "}\n";

        std::string code = src.str();
        {
            std::lock_guard<std::mutex> lock(contig_mutex);
            if (contig_cache.count(code)) {
                kernel = contig_cache[code];
            } else {
                auto ptx = compile_ptx(code, cuDevice);
                CUmodule module;
                CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
                CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "contig_kernel"));
                contig_cache[code] = kernel;
            }
        }

        void* args[] = {&in_buf->ptr, &out_buffer->ptr};
        const int threads = get_elementwise_threads();
        int blocks = (elements + threads - 1) / threads;
        CUDA_SAFE_CALL(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, get_stream(), args, 0));

        return out_buffer;
    }

    void ORMI_cudaMemsetAsync(void* ptr, int value, size_t bytes) {
        CUDART_SAFE_CALL(cudaMemsetAsync(ptr, value, bytes, get_stream()));
    }
}