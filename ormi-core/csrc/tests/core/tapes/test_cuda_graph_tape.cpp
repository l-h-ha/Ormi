#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <functional>

#include "tests/test_utils.h"
#include "core/device.h"
#include "core/allocator.h"

#include <cuda_runtime.h>
#include "core/tapes/cuda_graph_tape.cuh"
#include "backend/cuda/utils.cuh"

using namespace ormi;

class CUDAGraphTapeTest : public ormi::testing::HardwareTestSuite {
protected:
    void SetUp() override {
        HardwareTestSuite::SetUp();
        bool has_cuda = false;
        for (auto dev : available_devices) {
            if (dev.type == DeviceEnum::CUDA) {
                has_cuda = true;
            }
        }
        if (!has_cuda) {
            GTEST_SKIP() << "Skipping CUDA Graph tests: No active CUDA hardware detected.";
        }
    }
};

TEST_F(CUDAGraphTapeTest, CaptureAndLaunchExecution) {
    // Allocate 4 bytes
    int* d_ptr = nullptr;
    cudaMalloc(&d_ptr, sizeof(int));
    
    // Seed host verification data
    int initial_val = 55;
    int host_verify = 0;
    cudaMemcpy(d_ptr, &initial_val, sizeof(int), cudaMemcpyHostToDevice);

    CudaGraphTape tape;
    tape.capture([&]() {
        cudaMemsetAsync(d_ptr, 0, sizeof(int), cuda::get_stream());
    });

    // Modify value back to check if tape execution actually resets it
    cudaMemcpy(d_ptr, &initial_val, sizeof(int), cudaMemcpyHostToDevice);

    // Replay captured operations via graph invocation
    tape.execute();
    cudaStreamSynchronize(cuda::get_stream());

    cudaMemcpy(&host_verify, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host_verify, 0);
    cudaFree(d_ptr);
}

TEST_F(CUDAGraphTapeTest, GraphReplayAndExecUpdateSuccess) {
    int* d_ptr = nullptr;
    cudaMalloc(&d_ptr, sizeof(int));

    CudaGraphTape tape;
    tape.capture([&]() {
        cudaMemsetAsync(d_ptr, 0, sizeof(int), cuda::get_stream());
    });

    // matching topology triggers the optimized cudaGraphExecUpdate path
    EXPECT_NO_THROW({
        tape.capture([&]() {
            cudaMemsetAsync(d_ptr, 0, sizeof(int), cuda::get_stream());
        });
    });

    cudaFree(d_ptr);
}

TEST_F(CUDAGraphTapeTest, GraphTopologyChangeTriggersFallbackReinstantiation) {
    int* d_ptr_1 = nullptr;
    int* d_ptr_2 = nullptr;
    cudaMalloc(&d_ptr_1, sizeof(int));
    cudaMalloc(&d_ptr_2, sizeof(int));

    CudaGraphTape tape;

    // single-node graph topology
    tape.capture([&]() {
        cudaMemsetAsync(d_ptr_1, 0, sizeof(int), cuda::get_stream());
    });

    // change in topology (adding an extra dependency node)
    // This forces cudaGraphExecUpdate to fail and safely exercise the fallback branch
    EXPECT_NO_THROW({
        tape.capture([&]() {
            cudaMemsetAsync(d_ptr_1, 0, sizeof(int), cuda::get_stream());
            cudaMemsetAsync(d_ptr_2, 0, sizeof(int), cuda::get_stream());
        });
    });

    cudaFree(d_ptr_1);
    cudaFree(d_ptr_2);
}

TEST_F(CUDAGraphTapeTest, UninitializedExecuteBehavesAsNoOp) {
    CudaGraphTape tape;
    
    // Calling execute before capture must early exit
    EXPECT_NO_THROW(tape.execute());
}