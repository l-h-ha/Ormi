#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cstring>
#include <cuda_runtime.h>
#include <half.hpp>

#include "tests/test_utils.h"
#include "core/buffer.h"
#include "core/dtype.h"
#include "core/allocator.h"
#include "backend/cuda/optimizer.cuh"
#include "backend/cuda/utils.cuh"

using namespace ormi;

class CUDAOptimizerTest : public ormi::testing::HardwareTestSuite {
protected:
    void SetUp() override {
        HardwareTestSuite::SetUp();
        bool has_cuda = false;
        for (auto dev : available_devices) {
            if (dev.type == DeviceEnum::CUDA) has_cuda = true;
        }
        if (!has_cuda) {
            GTEST_SKIP() << "Skipping CUDA Optimizer tests: No active CUDA hardware detected.";
        }
    }

    std::shared_ptr<Buffer> make_buffer(int size, float fill_val, DType dtype) {
        return full({size}, fill_val, Device::cuda(), dtype);
    }
};

TEST_F(CUDAOptimizerTest, SGDFloat32Execution) {
    int size = 1024; 
    float initial_weight = 2.0f;
    float gradient = 0.5f;
    float lr = 0.1f;

    auto weight = make_buffer(size, initial_weight, DType::Float32);
    auto grad = make_buffer(size, gradient, DType::Float32);

    cuda::sgd_step(grad, weight, lr);
    cudaStreamSynchronize(cuda::get_stream());

    std::vector<float> h_weight(size);
    IAllocator* allocator = get_allocator(Device::cuda(0));
    allocator->copy_device_to_host(h_weight.data(), weight->ptr, weight->bytes);

    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(h_weight[i], 1.95f);
    }
}

TEST_F(CUDAOptimizerTest, SGDFloat16Execution) {
    int size = 512;
    float initial_weight = 4.0f; 
    float gradient = 2.0f;
    float lr = 0.5f;

    auto weight = make_buffer(size, initial_weight, DType::Float16);
    auto grad = make_buffer(size, gradient, DType::Float16);

    cuda::sgd_step(grad, weight, lr);
    cudaStreamSynchronize(cuda::get_stream());

    std::vector<unsigned short> h_weight_bits(size);
    IAllocator* allocator = get_allocator(Device::cuda(0));
    allocator->copy_device_to_host(h_weight_bits.data(), weight->ptr, weight->bytes);

    // 4.0 - (2.0 * 0.5) = 3.0
    for (int i = 0; i < size; ++i) {
        unsigned short raw_bits = h_weight_bits[i];
        half_float::half h;
        std::memcpy(&h, &raw_bits, sizeof(unsigned short));
        float result = static_cast<float>(h);
        
        EXPECT_FLOAT_EQ(result, 3.0f);
    }
}

TEST_F(CUDAOptimizerTest, UnrecognizedDTypeThrowsError) {
    int size = 10;
    auto weight = make_buffer(size, 1.0f, DType::Int32);
    auto grad = make_buffer(size, 1.0f, DType::Int32);

    EXPECT_THROW(
        cuda::sgd_step(grad, weight, 0.01f), 
        ormi::DTypeError
    );
}

TEST_F(CUDAOptimizerTest, MemoryAlignmentAndLargeBounds) {
    int size = 10007; 
    float initial_weight = 1.0f;
    float gradient = 1.0f;
    float lr = 1.0f;

    auto weight = make_buffer(size, initial_weight, DType::Float32);
    auto grad = make_buffer(size, gradient, DType::Float32);

    cuda::sgd_step(grad, weight, lr);
    cudaStreamSynchronize(cuda::get_stream());

    std::vector<float> h_weight(size);
    IAllocator* allocator = get_allocator(Device::cuda(0));
    allocator->copy_device_to_host(h_weight.data(), weight->ptr, weight->bytes);

    // 1.0 - (1.0 * 1.0) = 0.0
    EXPECT_FLOAT_EQ(h_weight[size - 1], 0.0f);
}