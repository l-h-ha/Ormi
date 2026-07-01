#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <memory>

#include "tests/test_utils.h"
#include "core/buffer.h"
#include "core/float16.h"
#include "core/node.h"
#include "core/shape.h"
#include "backend/cuda/kernels/blas/matmul.cuh"

using namespace ormi;

class CUDAMatMulKernelTest : public ormi::testing::HardwareTestSuite {
protected:
    void SetUp() override {
        HardwareTestSuite::SetUp();
        bool has_cuda = false;
        for (auto dev : available_devices) {
            if (dev.type == DeviceEnum::CUDA) has_cuda = true;
        }
        if (!has_cuda) GTEST_SKIP() << "Skipping CUDA MatMul tests: CUDA device unavailable.";
    }

    std::shared_ptr<Buffer> make_buffer(const std::vector<float>& data) {
        auto dev = Device(DeviceEnum::CUDA);
        auto buf = empty({(int)data.size()}, dev, DType::Float32);
        get_allocator(dev)->copy_host_to_device(buf->ptr, data.data(), data.size() * sizeof(float));
        return buf;
    }

    TensorView make_view(const std::vector<int>& shape) {
        TensorView v;
        v.shape = shape;
        v.strides = calc_contiguous_strides(shape);
        return v;
    }
};

TEST_F(CUDAMatMulKernelTest, Basic2D) {
    auto a_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto a_view = make_view({2, 3});

    auto b_buf = make_buffer({7, 8, 9, 10, 11, 12});
    auto b_view = make_view({3, 2});

    auto c_view = make_view({2, 2});

    auto c_buf = cuda::matmul(a_buf, a_view, b_buf, b_view, c_view);

    std::vector<float> out(4);
    get_allocator(Device::cuda(0))->copy_device_to_host(out.data(), c_buf->ptr, 4 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 58.0f);
    EXPECT_FLOAT_EQ(out[1], 64.0f);
    EXPECT_FLOAT_EQ(out[2], 139.0f);
    EXPECT_FLOAT_EQ(out[3], 154.0f);
}

TEST_F(CUDAMatMulKernelTest, Transposed) {
    auto a_buf = make_buffer({1, 2, 3, 4, 5, 6});
    TensorView a_view;
    a_view.shape = {3, 2};
    a_view.strides = {1, 3};

    auto b_buf = make_buffer({1, 2, 3, 4});
    auto b_view = make_view({2, 2});

    auto c_view = make_view({3, 2});

    auto c_buf = cuda::matmul(a_buf, a_view, b_buf, b_view, c_view);

    std::vector<float> out(6);
    get_allocator(Device::cuda(0))->copy_device_to_host(out.data(), c_buf->ptr, 6 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 13.0f);
    EXPECT_FLOAT_EQ(out[1], 18.0f);
    EXPECT_FLOAT_EQ(out[2], 17.0f);
    EXPECT_FLOAT_EQ(out[3], 24.0f);
    EXPECT_FLOAT_EQ(out[4], 21.0f);
    EXPECT_FLOAT_EQ(out[5], 30.0f);
}

TEST_F(CUDAMatMulKernelTest, BatchedBroadcasting) {
    auto a_buf = make_buffer({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto a_view = make_view({2, 2, 3});

    auto b_buf = make_buffer({1, 2, 3, 4, 5, 6});
    TensorView b_view;
    b_view.shape = {2, 3, 2};
    b_view.strides = {0, 2, 1};

    auto c_view = make_view({2, 2, 2});

    auto c_buf = cuda::matmul(a_buf, a_view, b_buf, b_view, c_view);

    std::vector<float> out(8);
    get_allocator(Device::cuda(0))->copy_device_to_host(out.data(), c_buf->ptr, 8 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 22.0f);
    EXPECT_FLOAT_EQ(out[1], 28.0f);
    EXPECT_FLOAT_EQ(out[2], 49.0f);
    EXPECT_FLOAT_EQ(out[3], 64.0f);

    EXPECT_FLOAT_EQ(out[4], 76.0f);
    EXPECT_FLOAT_EQ(out[5], 100.0f);
    EXPECT_FLOAT_EQ(out[6], 103.0f);
    EXPECT_FLOAT_EQ(out[7], 136.0f);
}

TEST_F(CUDAMatMulKernelTest, Float16Execution) {
    auto dev = Device(DeviceEnum::CUDA);
    auto alloc = get_allocator(dev);

    // 2x2 identity matrix @ 2x2 matrix
    std::vector<float> a_data = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> b_data = {2.0f, 3.0f, 4.0f, 5.0f};

    auto make_fp16_buf = [&](const std::vector<float>& host_floats) {
        auto buf = empty({(int)host_floats.size()}, dev, DType::Float16);
        std::vector<std::uint16_t> fp16_bits(host_floats.size());
        for (size_t i = 0; i < host_floats.size(); ++i) {
            fp16_bits[i] = fp16::float_to_bits(host_floats[i]);
        }
        alloc->copy_host_to_device(buf->ptr, fp16_bits.data(), fp16_bits.size() * sizeof(std::uint16_t));
        return buf;
    };

    auto a_buf = make_fp16_buf(a_data);
    auto b_buf = make_fp16_buf(b_data);

    auto view = make_view({2, 2});
    auto c_buf = cuda::matmul(a_buf, view, b_buf, view, view);

    // Read back bits
    std::vector<std::uint16_t> out_bits(4);
    alloc->copy_device_to_host(out_bits.data(), c_buf->ptr, 4 * sizeof(std::uint16_t));

    // identical to b_data since A is the identity matrix
    std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(fp16::bits_to_float(out_bits[i]), expected[i]);
    }
}
