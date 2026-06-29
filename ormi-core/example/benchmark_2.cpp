#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cblas.h>

#include "core/allocator.h"
#include "core/autograd.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/node_allocator.h"
#include "core/ops/blas/matmul.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/reduce/sum.h"
#include "core/optim/sgd.h"
#include "core/tapes/compute_tape.h"

#include "backend/runtime.h"

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

    double measure_latency_us(
        const Device& device,
        const std::function<void()>& func
    ) {
        sync_device(device);

        auto start = std::chrono::high_resolution_clock::now();

        func();

        sync_device(device);

        auto end = std::chrono::high_resolution_clock::now();

        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
            ).count()
        );
    }

    void print_buffer_summary(
        const char* name,
        const std::shared_ptr<Buffer>& buffer
    ) {
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
            if (i) {
                std::cout << ", ";
            }

            std::cout << buffer->shape[i];
        }

        std::cout << "]\n";
    }

    float first_value(const std::shared_ptr<Buffer>& buffer) {
        float value = 0.0f;

        get_allocator(buffer->device)->copy_device_to_host(
            &value,
            buffer->ptr,
            sizeof(float)
        );

        return value;
    }

    void print_build_info() {
        std::cout << "\nCPU Benchmark Info:\n";

    #ifdef NDEBUG
        std::cout << "  Build Type        : Release\n";
    #else
        std::cout << "  Build Type        : Debug\n";
    #endif

        std::cout << "  Device            : CPU\n\n";
    }
}

double benchmark_raw_cblas_training_gemms(
    const Device& device,
    const std::vector<int>& shape,
    int epochs
) {
    const int batch = shape[0];
    const int M = shape[1];
    const int K = shape[2];
    const int N = shape[2];

    auto A_buf = full(shape, 0.01f, device, DType::Float32);
    auto B_buf = full(shape, 0.02f, device, DType::Float32);
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

    auto run_one_epoch = [&]() {
        for (int b = 0; b < batch; ++b) {
            const float* A_b = A + b * M * K;
            const float* B_b = B + b * K * N;
            const float* dC_b = dC + b * M * N;

            float* C_b = C + b * M * N;
            float* dA_b = dA + b * M * K;
            float* dB_b = dB + b * K * N;

            // C = A @ B
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                1.0f,
                A_b,
                K,
                B_b,
                N,
                0.0f,
                C_b,
                N
            );

            // dA = dC @ B^T
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                M,
                K,
                N,
                1.0f,
                dC_b,
                N,
                B_b,
                N,
                0.0f,
                dA_b,
                K
            );

            // dB = A^T @ dC
            cblas_sgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                K,
                N,
                M,
                1.0f,
                A_b,
                K,
                dC_b,
                N,
                0.0f,
                dB_b,
                N
            );
        }
    };

    run_one_epoch();
    sync_device(device);

    double total_us = measure_latency_us(device, [&]() {
        for (int i = 0; i < epochs; ++i) {
            run_one_epoch();
        }
    });

    return total_us / static_cast<double>(epochs);
}

int main2() {
    print_build_info();
    std::cout << std::setprecision(9);

    const Device device = Device::cpu();
    const DType dtype = DType::Float32;
    const std::vector<int> shape = {4, 512, 512};
    const int epochs = 20;
    
    const int warmup_epochs = 5;
    const float learning_rate = 1e-6f;

    std::cout << "Benchmarking CPU training for "
              << epochs << " epochs...\n";

    {
        auto A_buf = full(shape, 0.01f, device, dtype);
        auto B_buf = full(shape, 0.02f, device, dtype);

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

        std::cout << "\n[1/4] Building graph...\n";

        auto C = ops::matmul(A, B);
        C->set_name("C");

        auto C_squared = ops::mul(C, C);
        C_squared->set_name("C_squared");

        auto loss = ops::sum(C_squared);
        loss->set_name("loss");

        auto grads = grad(loss, {A, B});
        auto A_grad_expr = grads[0];
        auto B_grad_expr = grads[1];

        optim::SGD optimizer({
            {A, A_grad_expr},
            {B, B_grad_expr}
        }, learning_rate);

        ComputeTape compute_tape;
        compute_tape.capture({
            loss,
            A_grad_expr,
            B_grad_expr
        });

        std::cout << "      -> Graph built\n\n";

        std::cout << "[2/4] Realizing buffers...\n";

        compute_tape.execute();
        sync_device(device);

        std::cout << "Graph buffers:\n";
        print_buffer_summary("A", A->data);
        print_buffer_summary("B", B->data);
        print_buffer_summary("C", C->data);
        print_buffer_summary("C_squared", C_squared->data);
        print_buffer_summary("loss", loss->data);
        print_buffer_summary("A_grad_expr", A_grad_expr->data);
        print_buffer_summary("B_grad_expr", B_grad_expr->data);
        std::cout << "\n";

        std::cout << "Initial state:"
                  << " LossRaw=" << first_value(loss->data)
                  << " A[0]=" << first_value(A->data)
                  << " B[0]=" << first_value(B->data)
                  << " A_grad_expr[0]=" << first_value(A_grad_expr->data)
                  << " B_grad_expr[0]=" << first_value(B_grad_expr->data)
                  << "\n\n";

        std::cout << "[3/4] Warming up...\n";

        double warmup_total_us = measure_latency_us(device, [&]() {
            for (int i = 0; i < warmup_epochs; ++i) {
                compute_tape.execute();
                optimizer.step();
            }
        });

        std::cout << "      -> Warmup Avg Epoch: "
                  << warmup_total_us / static_cast<double>(warmup_epochs)
                  << " us\n\n";

        std::cout << "[4/4] CPU Eager Benchmark...\n";

        double compute_total_us = measure_latency_us(device, [&]() {
            for (int i = 0; i < epochs; ++i) {
                compute_tape.execute();
            }
        });

        double optim_total_us = measure_latency_us(device, [&]() {
            for (int i = 0; i < epochs; ++i) {
                optimizer.step();
            }
        });

        double train_total_us = measure_latency_us(device, [&]() {
            for (int i = 0; i < epochs; ++i) {
                compute_tape.execute();
                optimizer.step();
            }
        });

        const double avg_compute_us =
            compute_total_us / static_cast<double>(epochs);

        const double avg_optim_us =
            optim_total_us / static_cast<double>(epochs);

        const double avg_train_us =
            train_total_us / static_cast<double>(epochs);

        compute_tape.execute();
        sync_device(device);

        std::cout << "Final state:"
                  << " LossRaw=" << first_value(loss->data)
                  << " A[0]=" << first_value(A->data)
                  << " B[0]=" << first_value(B->data)
                  << " A_grad_expr[0]=" << first_value(A_grad_expr->data)
                  << " B_grad_expr[0]=" << first_value(B_grad_expr->data)
                  << "\n\n";

        double raw_cblas_epoch_us = benchmark_raw_cblas_training_gemms(
            device,
            shape,
            epochs
        );

        std::cout << "  Raw CBLAS Epoch   : " << raw_cblas_epoch_us << " us\n";
        std::cout << "  ORMI/CBLAS        : " << avg_train_us / raw_cblas_epoch_us << "x\n";

        const long long batch = shape[0];
        const long long M = shape[1];
        const long long K = shape[2];
        const long long N = shape[2];

        const double single_matmul_flops =
            2.0 * static_cast<double>(batch) * M * N * K;

        const double total_fwd_bwd_flops =
            single_matmul_flops * 3.0;

        const double avg_train_sec =
            avg_train_us / 1'000'000.0;

        const double cpu_gflops =
            (total_fwd_bwd_flops / 1e9) / avg_train_sec;

        size_t total_memory_footprint = 0;
        total_memory_footprint += A_buf->bytes;
        total_memory_footprint += B_buf->bytes;

        if (C->data) {
            total_memory_footprint += C->data->bytes;
        }

        if (C_squared->data) {
            total_memory_footprint += C_squared->data->bytes;
        }

        if (loss->data) {
            total_memory_footprint += loss->data->bytes;
        }

        if (A_grad_expr->data) {
            total_memory_footprint += A_grad_expr->data->bytes;
        }

        if (B_grad_expr->data) {
            total_memory_footprint += B_grad_expr->data->bytes;
        }

        std::cout << std::fixed << std::setprecision(2);

        std::cout << "Results:\n";
        std::cout << "  CPU Compute Epoch : " << avg_compute_us << " us\n";
        std::cout << "  CPU Optim Epoch   : " << avg_optim_us << " us\n";
        std::cout << "  CPU Train Epoch   : " << avg_train_us << " us\n";

        std::cout << "\nComputation Throughput:\n";
        std::cout << "  Per-epoch FLOPs   : "
                  << total_fwd_bwd_flops / 1e9
                  << " GFLOPs\n";

        std::cout << "  CPU Throughput    : "
                  << cpu_gflops
                  << " GFLOPs/sec\n";

        std::cout << "\nMemory Footprint:\n";
        std::cout << "  Approx Tensor Memory: "
                  << format_bytes(total_memory_footprint)
                  << "\n";

        std::cout << "==================================================\n";
    }

    sync_device(device);
    get_allocator(device)->empty_cache();

    return 0;
}