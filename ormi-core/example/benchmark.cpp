#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cblas.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "core/allocator.h"
#include "core/autograd.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/node_allocator.h"
#include "core/ops.h"
#include "core/ops/blas/matmul.h"
#include "core/ops/reduce/sum.h"
#include "core/optim/sgd.h"
#include "core/tapes/compute_tape.h"
#include "core/tapes/cuda_graph_tape.cuh"

#include "backend/cuda/config.cuh"
#include "backend/cuda/utils.cuh"
#include "backend/runtime.h"
#include "utils/macros.h"

using namespace ormi;

namespace {
    std::string format_bytes(size_t bytes) {
        double b = static_cast<double>(bytes);

        if (b >= 1024.0 * 1024.0 * 1024.0) {
            return std::to_string(b / (1024.0 * 1024.0 * 1024.0)).substr(0, 5) + " GB";
        }

        if (b >= 1024.0 * 1024.0) {
            return std::to_string(b / (1024.0 * 1024.0)).substr(0, 5) + " MB";
        }

        return std::to_string(b / 1024.0).substr(0, 5) + " KB";
    }

    void sync_device(const Device& device) {
        synchronize(device.type);
    }

    double measure_latency(const Device& device, const std::function<void()>& func) {
        sync_device(device);

        auto start = std::chrono::high_resolution_clock::now();

        func();

        sync_device(device);

        auto end = std::chrono::high_resolution_clock::now();

        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
        );
    }

    double benchmark_graph_warmed(
        const Device& device,
        CudaGraphTape& graph,
        int epochs,
        int warmup_iters = 10
    ) {
        measure_latency(device, [&]() {
            for (int i = 0; i < warmup_iters; ++i) {
                graph.execute();
            }
        });

        double total_us = measure_latency(device, [&]() {
            for (int i = 0; i < epochs; ++i) {
                graph.execute();
            }
        });

        return total_us / static_cast<double>(epochs);
    }

    void print_buffer_summary(const char* name, const std::shared_ptr<Buffer>& buffer) {
        if (!buffer) {
            std::cout << "  " << name << ": null\n";
            return;
        }

        std::cout << "  " << name
                << ": bytes=" << buffer->bytes
                << " (" << format_bytes(buffer->bytes) << ")"
                << ", device=" << buffer->device
                << ", dtype=" << buffer->dtype
                << ", shape=[";

        for (size_t i = 0; i < buffer->shape.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << buffer->shape[i];
        }

        std::cout << "]\n";
    }

    bool local_supports_tf32() {
        int device_id = 0;
        CUDART_SAFE_CALL(cudaGetDevice(&device_id));

        cudaDeviceProp prop{};
        CUDART_SAFE_CALL(cudaGetDeviceProperties(&prop, device_id));

        return prop.major >= 8;
    }

    cublasComputeType_t get_fp32_compute_type() {
        return local_supports_tf32()
            ? CUBLAS_COMPUTE_32F_FAST_TF32
            : CUBLAS_COMPUTE_32F;
    }

    void print_cuda_benchmark_info() {
        int device_id = 0;
        CUDART_SAFE_CALL(cudaGetDevice(&device_id));

        cudaDeviceProp prop{};
        CUDART_SAFE_CALL(cudaGetDeviceProperties(&prop, device_id));

        std::cout << "\nCUDA Benchmark Info:\n";

    #ifdef NDEBUG
        std::cout << "  Build Type        : Release\n";
    #else
        std::cout << "  Build Type        : Debug\n";
    #endif

        std::cout << "  Device ID          : " << device_id << "\n";
        std::cout << "  GPU                : " << prop.name << "\n";
        std::cout << "  Compute Capability : "
                << prop.major << "." << prop.minor << "\n";
        std::cout << "  SM Count           : " << prop.multiProcessorCount << "\n";
        std::cout << "  Global Memory      : "
                << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
                << " GB\n";
        std::cout << "  TF32 Supported     : " << (local_supports_tf32() ? "yes" : "no") << "\n";

        cublasHandle_t handle = ormi::cuda::get_cublas_handle();

        int version = 0;
        CUBLAS_SAFE_CALL(cublasGetVersion(handle, &version));
        std::cout << "  cuBLAS Version     : " << version << "\n";

        cublasMath_t math_mode;
        CUBLAS_SAFE_CALL(cublasGetMathMode(handle, &math_mode));
        std::cout << "  cuBLAS Math Mode   : " << static_cast<int>(math_mode) << "\n";

        cudaStream_t stream = ormi::cuda::get_stream();
        std::cout << "  ORMI Stream        : " << stream << "\n\n";
    }

    void cublas_row_major_batched_gemm_f32(
        const float* A,
        const float* B,
        float* C,
        int batch_count,
        int M,
        int K,
        int N
    ) {
        cublasHandle_t handle = ormi::cuda::get_cublas_handle();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const long long stride_A = static_cast<long long>(M) * K;
        const long long stride_B = static_cast<long long>(K) * N;
        const long long stride_C = static_cast<long long>(M) * N;

        CUBLAS_SAFE_CALL(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            B,
            CUDA_R_32F,
            N,
            stride_B,
            A,
            CUDA_R_32F,
            K,
            stride_A,
            &beta,
            C,
            CUDA_R_32F,
            N,
            stride_C,
            batch_count,
            get_fp32_compute_type(),
            CUBLAS_GEMM_DEFAULT
        ));
    }

    void cublas_row_major_batched_grad_a_f32(
        const float* dC,
        const float* B,
        float* dA,
        int batch_count,
        int M,
        int K,
        int N
    ) {
        cublasHandle_t handle = ormi::cuda::get_cublas_handle();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const long long stride_dC = static_cast<long long>(M) * N;
        const long long stride_B = static_cast<long long>(K) * N;
        const long long stride_dA = static_cast<long long>(M) * K;

        CUBLAS_SAFE_CALL(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            K,
            M,
            N,
            &alpha,
            B,
            CUDA_R_32F,
            N,
            stride_B,
            dC,
            CUDA_R_32F,
            N,
            stride_dC,
            &beta,
            dA,
            CUDA_R_32F,
            K,
            stride_dA,
            batch_count,
            get_fp32_compute_type(),
            CUBLAS_GEMM_DEFAULT
        ));
    }

    void cublas_row_major_batched_grad_b_f32(
        const float* A,
        const float* dC,
        float* dB,
        int batch_count,
        int M,
        int K,
        int N
    ) {
        cublasHandle_t handle = ormi::cuda::get_cublas_handle();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        const long long stride_A = static_cast<long long>(M) * K;
        const long long stride_dC = static_cast<long long>(M) * N;
        const long long stride_dB = static_cast<long long>(K) * N;

        CUBLAS_SAFE_CALL(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            N,
            K,
            M,
            &alpha,
            dC,
            CUDA_R_32F,
            N,
            stride_dC,
            A,
            CUDA_R_32F,
            K,
            stride_A,
            &beta,
            dB,
            CUDA_R_32F,
            N,
            stride_dB,
            batch_count,
            get_fp32_compute_type(),
            CUBLAS_GEMM_DEFAULT
        ));
    }

    double benchmark_raw_cublas_training_gemms(
        const Device& device,
        const std::vector<int>& shape,
        int epochs
    ) {
        const int batch = shape[0];
        const int M = shape[1];
        const int K = shape[2];
        const int N = shape[2];

        auto A_buf = full(shape, 0.01f, device, DType::Float32);
        auto B_buf = full(shape, 0.01f, device, DType::Float32);
        auto dC_buf = full(shape, 1.0f, device, DType::Float32);

        auto C_buf = empty(shape, device, DType::Float32);
        auto dA_buf = empty(shape, device, DType::Float32);
        auto dB_buf = empty(shape, device, DType::Float32);

        const float* A = static_cast<const float*>(A_buf->ptr);
        const float* B = static_cast<const float*>(B_buf->ptr);
        const float* dC = static_cast<const float*>(dC_buf->ptr);

        float* C = static_cast<float*>(C_buf->ptr);
        float* dA = static_cast<float*>(dA_buf->ptr);
        float* dB = static_cast<float*>(dB_buf->ptr);

        cublas_row_major_batched_gemm_f32(A, B, C, batch, M, K, N);
        cublas_row_major_batched_grad_a_f32(dC, B, dA, batch, M, K, N);
        cublas_row_major_batched_grad_b_f32(A, dC, dB, batch, M, K, N);
        sync_device(device);

        double raw_total_us = measure_latency(device, [&]() {
            for (int i = 0; i < epochs; ++i) {
                cublas_row_major_batched_gemm_f32(A, B, C, batch, M, K, N);
                cublas_row_major_batched_grad_a_f32(dC, B, dA, batch, M, K, N);
                cublas_row_major_batched_grad_b_f32(A, dC, dB, batch, M, K, N);
            }
        });

        return raw_total_us / static_cast<double>(epochs);
    }

}

int main5() {
    print_cuda_benchmark_info();

    const Device device = Device::cuda(0);
    const DType dtype = DType::Float32;

    const std::vector<int> shape = {4, 512, 512};
    const int epochs = 200;
    const float learning_rate = 1e-3f;

    ormi::cuda::CudaConfig config = ormi::cuda::get_config();
    config.optimizer_threads = 128;
    ormi::cuda::set_config(config);

    std::cout << "  CUDA Config       : " << ormi::cuda::get_config() << "\n";
    std::cout << "Benchmarking for " << epochs << " epochs...\n";

    {
        auto A_buf = full(shape, 0.01f, device, dtype);
        auto B_buf = full(shape, 0.01f, device, dtype);

        auto A = std::allocate_shared<Node>(
            get_node_allocator<Node>(),
            A_buf,
            true,
            "A"
        );

        auto B = std::allocate_shared<Node>(
            get_node_allocator<Node>(),
            B_buf,
            true,
            "B"
        );

        std::cout << "\n[1/5] Warming up...\n";

        double warmup_us = measure_latency(device, [&]() {
            auto warm_C = ops::matmul(A, B);
            auto warm_loss = ops::sum(warm_C);
            auto warm_grads = grad(warm_loss, {A, B});

            ComputeTape warm_tape;
            warm_tape.capture({warm_loss, warm_grads[0], warm_grads[1]});
            warm_tape.execute();
        });

        std::cout << "      -> Warmup Latency: " << warmup_us << " us\n\n";

        auto C = ops::matmul(A, B);
        auto loss = ops::sum(C);

        auto grads = grad(loss, {A, B});
        auto A_grad_expr = grads[0];
        auto B_grad_expr = grads[1];

        optim::SGD optimizer({
            {A, A_grad_expr},
            {B, B_grad_expr}
        }, learning_rate);

        std::vector<std::shared_ptr<Node>> compute_roots = {
            loss,
            A_grad_expr,
            B_grad_expr
        };

        ComputeTape compute_tape;
        compute_tape.capture(compute_roots);
        
        compute_tape.execute();
        sync_device(device);

        std::cout << "Graph buffers:\n";
        print_buffer_summary("A", A->data);
        print_buffer_summary("B", B->data);
        print_buffer_summary("A_grad_expr", A_grad_expr->data);
        print_buffer_summary("B_grad_expr", B_grad_expr->data);
        print_buffer_summary("C", C->data);
        print_buffer_summary("loss", loss->data);
        std::cout << "\n";

        std::cout << "[2/5] Eager Mode...\n";

        double eager_fwd_bwd_us = measure_latency(device, [&]() {
            compute_tape.execute();
        });

        double eager_optim_us = measure_latency(device, [&]() {
            optimizer.step();
        });

        double eager_total_us = eager_fwd_bwd_us + eager_optim_us;

        std::cout << "      -> Fwd/Bwd   : " << eager_fwd_bwd_us << " us\n";
        std::cout << "      -> Optimizer : " << eager_optim_us << " us\n";
        std::cout << "      -> Total Eager Step: " << eager_total_us << " us\n\n";

        std::cout << "[3/5] CUDA Graph Capture...\n";

        CudaGraphTape train_graph;

        double capture_us = measure_latency(device, [&]() {
            train_graph.capture([&]() {
                compute_tape.execute();
                optimizer.step();
            });
        });

        std::cout << "      -> Graph Capture Time: " << capture_us << " us\n";
        std::cout << "      -> Full graph nodes  : " << train_graph.node_count() << "\n\n";

        std::cout << "[4/5] CUDA Graph Benchmark...\n";

        double graph_warm_loop_us = measure_latency(device, [&]() {
            for (int i = 0; i < epochs; ++i) {
                train_graph.execute();
            }
        });

        double graph_loop_us = measure_latency(device, [&]() {
            for (int i = 0; i < epochs; ++i) {
                train_graph.execute();
            }
        });

        double cold_graph_epoch_us =
            graph_warm_loop_us / static_cast<double>(epochs);

        double avg_graph_epoch_us =
            graph_loop_us / static_cast<double>(epochs);

        std::cout << "      -> Cold Graph Epoch: "
                  << cold_graph_epoch_us << " us\n";

        std::cout << "      -> Graph Epoch     : "
                  << avg_graph_epoch_us << " us\n\n";

        std::cout << "[5/5] Raw cuBLAS Baseline...\n";

        double raw_cublas_epoch_us = benchmark_raw_cublas_training_gemms(
            device,
            shape,
            epochs
        );

        std::cout << "      -> Raw cuBLAS Epoch: "
                  << raw_cublas_epoch_us << " us\n";

        std::cout << "\n[extra] Compute-only CUDA Graph Benchmark...\n";

        CudaGraphTape compute_only_graph;

        double compute_only_capture_us = measure_latency(device, [&]() {
            compute_only_graph.capture([&]() {
                compute_tape.execute();
            });
        });

        std::cout << "      -> Compute-only nodes: "
                  << compute_only_graph.node_count() << "\n";

        double compute_only_epoch_us = benchmark_graph_warmed(
            device,
            compute_only_graph,
            epochs
        );

        std::cout << "      -> Compute-only Capture: "
                  << compute_only_capture_us << " us\n";

        std::cout << "      -> Compute-only Epoch  : "
                  << compute_only_epoch_us << " us\n\n";

        const long long batch = shape[0];
        const long long M = shape[1];
        const long long K = shape[2];
        const long long N = shape[2];

        const double single_matmul_flops =
            2.0 * static_cast<double>(batch) * M * N * K;

        const double total_fwd_bwd_flops = single_matmul_flops * 3.0;

        size_t total_memory_footprint = 0;
        total_memory_footprint += A_buf->bytes;
        total_memory_footprint += B_buf->bytes;

        if (A_grad_expr->data) {
            total_memory_footprint += A_grad_expr->data->bytes;
        }

        if (B_grad_expr->data) {
            total_memory_footprint += B_grad_expr->data->bytes;
        }

        if (C->data) {
            total_memory_footprint += C->data->bytes;
        }

        if (loss->data) {
            total_memory_footprint += loss->data->bytes;
        }

        const double avg_graph_epoch_sec = avg_graph_epoch_us / 1'000'000.0;
        const double raw_cublas_epoch_sec = raw_cublas_epoch_us / 1'000'000.0;

        const double graph_tflops =
            (total_fwd_bwd_flops / 1e12) / avg_graph_epoch_sec;

        const double raw_cublas_tflops =
            (total_fwd_bwd_flops / 1e12) / raw_cublas_epoch_sec;

        std::cout << std::fixed << std::setprecision(2);

        std::cout << "\nResults:\n";
        std::cout << "  Eager Mode Epoch : " << eager_total_us << " us\n";
        std::cout << "  Graph Mode Epoch : " << avg_graph_epoch_us << " us\n";
        std::cout << "  Raw cuBLAS Epoch : " << raw_cublas_epoch_us << " us\n";
        std::cout << "  Graph Speedup    : " << eager_total_us / avg_graph_epoch_us << "x\n";
        std::cout << "  Graph/cuBLAS     : " << avg_graph_epoch_us / raw_cublas_epoch_us << "x\n";

        std::cout << "\nComputation Throughput:\n";
        std::cout << "  Per-epoch FLOPs      : " << total_fwd_bwd_flops / 1e9 << " GFLOPs\n";
        std::cout << "  Total FLOPs          : " << (total_fwd_bwd_flops * epochs) / 1e9 << " GFLOPs\n";
        std::cout << "  Graph Throughput     : " << graph_tflops << " TFLOPs/sec\n";
        std::cout << "  Raw cuBLAS Throughput: " << raw_cublas_tflops << " TFLOPs/sec\n";

        std::cout << "\nMemory Footprint:\n";
        std::cout << "  Approx Tensor Memory: " << format_bytes(total_memory_footprint) << "\n";

        std::cout << "==================================================\n";
    }

    sync_device(device);
    get_allocator(device)->empty_cache();

    return 0;
}