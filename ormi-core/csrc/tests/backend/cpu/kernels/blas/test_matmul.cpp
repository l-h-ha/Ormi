#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "core/buffer.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/allocator.h"
#include "backend/cpu/kernels/blas/matmul.h"
#include "backend/graph_breakers.h"

using namespace ormi;

class CPUMatMulKernelTest : public ::testing::Test {
protected:
    std::shared_ptr<Buffer> make_buffer(const std::vector<float>& data) {
        auto dev = Device::cpu();
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

TEST_F(CPUMatMulKernelTest, Basic2D) {
    auto a_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto a_view = make_view({2, 3});

    auto b_buf = make_buffer({7, 8, 9, 10, 11, 12});
    auto b_view = make_view({3, 2});

    auto c_view = make_view({2, 2});

    auto c_buf = cpu::matmul(a_buf, a_view, b_buf, b_view, c_view);

    std::vector<float> out(4);
    get_allocator(Device::cpu())->copy_device_to_host(out.data(), c_buf->ptr, 4 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 58.0f);
    EXPECT_FLOAT_EQ(out[1], 64.0f);
    EXPECT_FLOAT_EQ(out[2], 139.0f);
    EXPECT_FLOAT_EQ(out[3], 154.0f);
}

TEST_F(CPUMatMulKernelTest, Transposed) {
    auto a_buf = make_buffer({1, 2, 3, 4, 5, 6});
    TensorView a_view;
    a_view.shape = {3, 2};
    a_view.strides = {1, 3};

    auto b_buf = make_buffer({1, 2, 3, 4});
    auto b_view = make_view({2, 2});

    auto c_view = make_view({3, 2});

    auto c_buf = cpu::matmul(a_buf, a_view, b_buf, b_view, c_view);

    std::vector<float> out(6);
    get_allocator(Device::cpu())->copy_device_to_host(out.data(), c_buf->ptr, 6 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 13.0f);
    EXPECT_FLOAT_EQ(out[1], 18.0f);
    EXPECT_FLOAT_EQ(out[2], 17.0f);
    EXPECT_FLOAT_EQ(out[3], 24.0f);
    EXPECT_FLOAT_EQ(out[4], 21.0f);
    EXPECT_FLOAT_EQ(out[5], 30.0f);
}

TEST_F(CPUMatMulKernelTest, BatchedBroadcasting) {
    auto a_buf = make_buffer({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto a_view = make_view({2, 2, 3});

    auto b_buf = make_buffer({1, 2, 3, 4, 5, 6});
    TensorView b_view;
    b_view.shape = {2, 3, 2};
    b_view.strides = {0, 2, 1};

    auto c_view = make_view({2, 2, 2});

    auto c_buf = cpu::matmul(a_buf, a_view, b_buf, b_view, c_view);

    std::vector<float> out(8);
    get_allocator(Device::cpu())->copy_device_to_host(out.data(), c_buf->ptr, 8 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 22.0f);
    EXPECT_FLOAT_EQ(out[1], 28.0f);
    EXPECT_FLOAT_EQ(out[2], 49.0f);
    EXPECT_FLOAT_EQ(out[3], 64.0f);

    EXPECT_FLOAT_EQ(out[4], 76.0f);
    EXPECT_FLOAT_EQ(out[5], 100.0f);
    EXPECT_FLOAT_EQ(out[6], 103.0f);
    EXPECT_FLOAT_EQ(out[7], 136.0f);
}

TEST_F(CPUMatMulKernelTest, GraphBreakerInterceptionHandler) {
    auto dev = Device(DeviceEnum::CPU);

    // two non-contiguous parents
    auto a_base = empty({3, 2}, dev, DType::Float32);
    auto b_base = empty({3, 2}, dev, DType::Float32);

    // Fill with values
    float val_a = 2.0f;
    float val_b = 3.0f;
    get_allocator(dev)->fill_n(a_base->ptr, &val_a, 6, DType::Float32);
    get_allocator(dev)->fill_n(b_base->ptr, &val_b, 6, DType::Float32);

    auto node_a = std::make_shared<Node>(a_base, false);
    // Force a transpose view layout change on parent A
    node_a->view.shape = {2, 3};
    node_a->view.strides = {1, 2};

    auto node_b = std::make_shared<Node>(b_base, false); // {3, 2}

    // MATMUL output Node
    auto root = std::make_shared<Node>(ops::OpEnum::MATMUL, std::vector<std::shared_ptr<Node>>{node_a, node_b});
    root->device = dev;
    root->view.shape = {2, 2};
    root->view.strides = calc_contiguous_strides(root->view.shape);

    bool intercepted = try_execute_graph_break(root);
    ASSERT_TRUE(intercepted);
    ASSERT_NE(root->data, nullptr);

    // [2, 3] @ [3, 2] -> [2, 2] containing (2 * 3) * 3 = 18.0f
    std::vector<float> out(4);
    get_allocator(dev)->copy_device_to_host(out.data(), root->data->ptr, 4 * sizeof(float));
    for (float val : out) {
        EXPECT_FLOAT_EQ(val, 18.0f);
    }
}