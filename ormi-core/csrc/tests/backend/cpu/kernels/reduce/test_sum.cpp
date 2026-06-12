#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "core/memory.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/allocator.h"
#include "backend/cpu/kernels/reduce/cpu_sum.h"

using namespace ormi;

class CPUSumKernelTest : public ::testing::Test {
protected:
    std::shared_ptr<Buffer> make_buffer(const std::vector<float>& data) {
        auto dev = Device(DeviceEnum::CPU);
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

TEST_F(CPUSumKernelTest, GlobalReduction) {
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto in_view = make_view({2, 3});
    auto out_view = make_view({}); 

    auto out_buf = cpu::sum(in_buf, in_view, {}, out_view, false);

    std::vector<float> out(1);
    get_allocator(Device(DeviceEnum::CPU).type)->copy_device_to_host(out.data(), out_buf->ptr, sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 21.0f);
}

TEST_F(CPUSumKernelTest, AxisReduction) {
    // [[1, 2, 3], 
    //  [4, 5, 6]]
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto in_view = make_view({2, 3});
    auto out_view = make_view({2});

    // Reduce axis 1 (columns)
    auto out_buf = cpu::sum(in_buf, in_view, {1}, out_view, false);

    std::vector<float> out(2);
    get_allocator(Device(DeviceEnum::CPU).type)->copy_device_to_host(out.data(), out_buf->ptr, 2 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 6.0f); 
    EXPECT_FLOAT_EQ(out[1], 15.0f);
}

TEST_F(CPUSumKernelTest, KeepDimsReduction) {
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto in_view = make_view({2, 3});
    auto out_view = make_view({1, 3});

    // Reduce axis 0 (rows)
    auto out_buf = cpu::sum(in_buf, in_view, {0}, out_view, true);

    std::vector<float> out(3);
    get_allocator(Device(DeviceEnum::CPU).type)->copy_device_to_host(out.data(), out_buf->ptr, 3 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 5.0f); 
    EXPECT_FLOAT_EQ(out[1], 7.0f); 
    EXPECT_FLOAT_EQ(out[2], 9.0f); 
}

TEST_F(CPUSumKernelTest, MultiAxisReduction) {
    // 2x2x2 cube: 1, 2, 3, 4, 5, 6, 7, 8
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6, 7, 8});
    auto in_view = make_view({2, 2, 2});
    auto out_view = make_view({2});

    // Reduce axis 0 and 2
    auto out_buf = cpu::sum(in_buf, in_view, {0, 2}, out_view, false);

    std::vector<float> out(2);
    get_allocator(Device(DeviceEnum::CPU).type)->copy_device_to_host(out.data(), out_buf->ptr, 2 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 14.0f); // 1 + 2 + 5 + 6
    EXPECT_FLOAT_EQ(out[1], 22.0f); // 3 + 4 + 7 + 8
}