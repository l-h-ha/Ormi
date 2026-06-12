#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "backend/cuda/jit.cuh"
#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/pointwise/sin.h"

using namespace ormi;

class CUDAJitTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, float val) {
        auto dev = Device(DeviceEnum::CUDA);
        auto buf = full(shape, val, dev.type);
        return std::make_shared<Node>(buf, false, dev);
    }

    void SetUp() override {
        HardwareTestSuite::SetUp();
        bool has_cuda = false;
        for (auto dev : available_devices) {
            if (dev == DeviceEnum::CUDA) has_cuda = true;
        }
        if (!has_cuda) GTEST_SKIP() << "Skipping CUDA JIT tests: CUDA device unavailable.";
    }
};

TEST_F(CUDAJitTest, BasicExecution) {
    auto a = make_leaf({4}, 2.0f);
    auto b = make_leaf({4}, 3.0f);
    auto c = ops::add(a, b);

    cuda::execute_jit(c);

    EXPECT_TRUE(c->is_realized());
    EXPECT_EQ(c->view.shape, (std::vector<int>{4}));

    std::vector<float> out(4);
    get_allocator(Device(DeviceEnum::CUDA).type)->copy_device_to_host(out.data(), c->data->ptr, 4 * sizeof(float));

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], 5.0f);
    }
}

TEST_F(CUDAJitTest, TapeChaining) {
    auto a = make_leaf({2, 2}, 2.0f);
    auto b = make_leaf({2, 2}, 3.0f);
    auto c = make_leaf({2, 2}, 4.0f);

    // D = (A + B) * C
    auto a_plus_b = ops::add(a, b);
    auto d = ops::mul(a_plus_b, c);

    cuda::execute_jit(d);

    std::vector<float> out(4);
    get_allocator(Device(DeviceEnum::CUDA).type)->copy_device_to_host(out.data(), d->data->ptr, 4 * sizeof(float));

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], 20.0f);
    }
}

TEST_F(CUDAJitTest, BroadcastingExecution) {
    // A: [2, 3] + B: [3] -> [2, 3]
    auto a = make_leaf({2, 3}, 2.0f);
    auto b = make_leaf({3}, 5.0f);
    auto c = ops::add(a, b);

    cuda::execute_jit(c);

    EXPECT_EQ(c->view.shape, (std::vector<int>{2, 3}));

    std::vector<float> out(6);
    get_allocator(Device(DeviceEnum::CUDA).type)->copy_device_to_host(out.data(), c->data->ptr, 6 * sizeof(float));

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out[i], 7.0f);
    }
}

TEST_F(CUDAJitTest, ScalarExecution) {
    // A: [2, 2] + B: [] -> [2, 2]
    auto a = make_leaf({2, 2}, 10.0f);
    auto b = make_leaf({}, 3.0f); 
    auto c = ops::add(a, b);

    cuda::execute_jit(c);

    std::vector<float> out(4);
    get_allocator(Device(DeviceEnum::CUDA).type)->copy_device_to_host(out.data(), c->data->ptr, 4 * sizeof(float));

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], 13.0f);
    }
}

TEST_F(CUDAJitTest, UnaryMathFunctions) {
    // PI / 2 approximation mapped to device
    auto a = make_leaf({2}, 1.570796327f); 
    auto b = ops::sin(a);

    cuda::execute_jit(b);

    std::vector<float> out(2);
    get_allocator(Device(DeviceEnum::CUDA).type)->copy_device_to_host(out.data(), b->data->ptr, 2 * sizeof(float));

    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(out[i], 1.0f, 1e-5f);
    }
}