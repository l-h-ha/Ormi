#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/memory.h"
#include "core/node.h"
#include "core/shape.h"
#include "backend/cuda/kernels/reduce/cuda_sum.cuh"

using namespace ormi;

class CUDASumKernelTest : public ormi::testing::HardwareTestSuite {
protected:
    void SetUp() override {
        HardwareTestSuite::SetUp();
        bool has_cuda = false;
        for (auto dev : available_devices) {
            if (dev == DeviceEnum::CUDA) has_cuda = true;
        }
        if (!has_cuda) GTEST_SKIP() << "Skipping CUDA Sum tests: CUDA device unavailable.";
    }

    std::shared_ptr<Buffer> make_buffer(const std::vector<float>& data) {
        auto dev = Device(DeviceEnum::CUDA);
        auto buf = empty({(int)data.size()}, dev.type);
        get_allocator(dev.type)->copy_host_to_device(buf->ptr, data.data(), data.size() * sizeof(float));
        return buf;
    }

    TensorView make_view(const std::vector<int>& shape) {
        TensorView v;
        v.shape = shape;
        v.strides = calc_contiguous_strides(shape);
        return v;
    }
};

TEST_F(CUDASumKernelTest, GlobalReduction) {
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto in_view = make_view({2, 3});
    auto out_view = make_view({}); 

    auto out_buf = cuda::sum(in_buf, in_view, {}, out_view, false);

    std::vector<float> out(1);
    get_allocator(DeviceEnum::CUDA)->copy_device_to_host(out.data(), out_buf->ptr, sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 21.0f);
}

TEST_F(CUDASumKernelTest, AxisReduction) {
    // [[1, 2, 3], 
    //  [4, 5, 6]]
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto in_view = make_view({2, 3});
    auto out_view = make_view({2});

    // Reduce axis 1 (columns)
    auto out_buf = cuda::sum(in_buf, in_view, {1}, out_view, false);

    std::vector<float> out(2);
    get_allocator(DeviceEnum::CUDA)->copy_device_to_host(out.data(), out_buf->ptr, 2 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 6.0f); 
    EXPECT_FLOAT_EQ(out[1], 15.0f);
}

TEST_F(CUDASumKernelTest, KeepDimsReduction) {
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto in_view = make_view({2, 3});
    auto out_view = make_view({1, 3});

    // Reduce axis 0 (rows)
    auto out_buf = cuda::sum(in_buf, in_view, {0}, out_view, true);

    std::vector<float> out(3);
    get_allocator(DeviceEnum::CUDA)->copy_device_to_host(out.data(), out_buf->ptr, 3 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 5.0f); 
    EXPECT_FLOAT_EQ(out[1], 7.0f); 
    EXPECT_FLOAT_EQ(out[2], 9.0f); 
}

TEST_F(CUDASumKernelTest, MultiAxisReduction) {
    // 2x2x2 cube: 1, 2, 3, 4, 5, 6, 7, 8
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6, 7, 8});
    auto in_view = make_view({2, 2, 2});
    auto out_view = make_view({2});

    // Reduce axis 0 and 2
    auto out_buf = cuda::sum(in_buf, in_view, {0, 2}, out_view, false);

    std::vector<float> out(2);
    get_allocator(DeviceEnum::CUDA)->copy_device_to_host(out.data(), out_buf->ptr, 2 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 14.0f); // 1 + 2 + 5 + 6
    EXPECT_FLOAT_EQ(out[1], 22.0f); // 3 + 4 + 7 + 8
}