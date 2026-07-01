#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/buffer.h"
#include "core/float16.h"
#include "core/node.h"
#include "core/shape.h"
#include "core/ops/reduce/sum.h"
#include "backend/cuda/kernels/reduce/sum.cuh"
#include "backend/graph_breakers.h"

using namespace ormi;

class CUDASumKernelTest : public ormi::testing::HardwareTestSuite {
protected:
    void SetUp() override {
        HardwareTestSuite::SetUp();
        bool has_cuda = false;
        for (auto dev : available_devices) {
            if (dev == Device::cuda(0)) has_cuda = true;
        }
        if (!has_cuda) GTEST_SKIP() << "Skipping CUDA Sum tests: CUDA device unavailable.";
    }

    std::shared_ptr<Buffer> make_buffer(const std::vector<float>& data) {
        auto dev = Device(Device::cuda(0));
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

TEST_F(CUDASumKernelTest, GlobalReduction) {
    auto in_buf = make_buffer({1, 2, 3, 4, 5, 6});
    auto in_view = make_view({2, 3});
    auto out_view = make_view({});

    auto out_buf = cuda::sum(in_buf, in_view, {}, out_view, false);

    std::vector<float> out(1);
    get_allocator(Device::cuda(0))->copy_device_to_host(out.data(), out_buf->ptr, sizeof(float));

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
    get_allocator(Device::cuda(0))->copy_device_to_host(out.data(), out_buf->ptr, 2 * sizeof(float));

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
    get_allocator(Device::cuda(0))->copy_device_to_host(out.data(), out_buf->ptr, 3 * sizeof(float));

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
    get_allocator(Device::cuda(0))->copy_device_to_host(out.data(), out_buf->ptr, 2 * sizeof(float));

    EXPECT_FLOAT_EQ(out[0], 14.0f); // 1 + 2 + 5 + 6
    EXPECT_FLOAT_EQ(out[1], 22.0f); // 3 + 4 + 7 + 8
}

TEST_F(CUDASumKernelTest, Float16GlobalReduction) {
    auto dev = Device(Device::cuda(0));
    auto alloc = get_allocator(dev);

    std::vector<float> in_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto in_buf = empty({4}, dev, DType::Float16);

    // Pack float to __half bits
    std::vector<std::uint16_t> fp16_bits(4);
    for (size_t i = 0; i < 4; ++i) {
        fp16_bits[i] = fp16::float_to_bits(in_data[i]);
    }
    alloc->copy_host_to_device(in_buf->ptr, fp16_bits.data(), 4 * sizeof(std::uint16_t));

    auto in_view = make_view({4});
    auto out_view = make_view({});

    auto out_buf = cuda::sum(in_buf, in_view, {}, out_view, false);

    std::uint16_t out_bit = 0;
    alloc->copy_device_to_host(&out_bit, out_buf->ptr, sizeof(std::uint16_t));

    // 1 + 2 + 3 + 4 = 10.0
    EXPECT_FLOAT_EQ(fp16::bits_to_float(out_bit), 10.0f);
}

TEST_F(CUDASumKernelTest, GraphBreakerAndBufferZeroing) {
    auto in_buf = make_buffer({1.0f, 2.0f, 3.0f, 4.0f});
    auto node_in = std::make_shared<Node>(in_buf, false);
    node_in->view = make_view({2, 2});

    auto root = ops::sum(node_in, {0}, false);
    root->data = make_buffer({99.0f, 99.0f}); // simulates garbage data

    bool intercepted = try_execute_graph_break(root);
    ASSERT_TRUE(intercepted);

    std::vector<float> out(2);
    get_allocator(Device::cuda(0))->copy_device_to_host(out.data(), root->data->ptr, 2 * sizeof(float));

    // [[1, 2], [3, 4]] -> [4, 6]
    // If the memset in sum_handler failed, this would equal [103.0, 105.0]
    EXPECT_FLOAT_EQ(out[0], 4.0f);
    EXPECT_FLOAT_EQ(out[1], 6.0f);
}
