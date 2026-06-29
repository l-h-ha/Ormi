#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/buffer.h"
#include "core/allocator.h"
#include "core/ops.h"
#include "backend/runtime.h"
#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/view/reshape.h"
#include "core/ops/blas/matmul.h"

using namespace ormi;

class BackendRuntimeTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, float fill_val, Device dev) {
        auto buf = full(shape, fill_val, dev, DType::Float32);
        return std::make_shared<Node>(buf, false);
    }
};

TEST_F(BackendRuntimeTest, DeepGraphTraversalAndJIT) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 2}, 2.0f, dev);
        auto b = make_leaf({2, 2}, 3.0f, dev);
        auto c = make_leaf({2, 2}, 4.0f, dev);

        // D = (A + B) * C
        auto a_plus_b = ops::add(a, b);
        auto d = ops::mul(a_plus_b, c);

        EXPECT_FALSE(a_plus_b->is_realized());
        EXPECT_FALSE(d->is_realized());

        // Trigger execution
        execute_graph(d);

        EXPECT_TRUE(a_plus_b->is_realized());
        EXPECT_TRUE(d->is_realized());

        // Verify mathematical execution ((2 + 3) * 4 = 20)
        std::vector<float> host_verify(4);
        IAllocator* allocator = get_allocator(dev);
        allocator->copy_device_to_host(host_verify.data(), d->data->ptr, d->data->bytes);

        for (float val : host_verify) {
            EXPECT_FLOAT_EQ(val, 20.0f);
        }
    }
}

TEST_F(BackendRuntimeTest, ViewOperationInterception) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, 1.0f, dev);

        // View ops should short-circuit the execution loop
        auto b = ops::reshape(a, {6, 4});
        execute_graph(b);

        EXPECT_TRUE(b->is_realized());

        // Pointers must be identical (zero-copy)
        EXPECT_EQ(b->data->ptr, a->data->ptr);
    }
}

TEST_F(BackendRuntimeTest, GraphBreakerExecution) {
    for (auto dev : available_devices) {
        // Matmul breaks the element-wise JIT tape
        // [2, 3] @ [3, 2] -> [2, 2]
        auto a = make_leaf({2, 3}, 2.0f, dev);
        auto b = make_leaf({3, 2}, 3.0f, dev);

        auto c = ops::matmul(a, b);
        execute_graph(c);

        EXPECT_TRUE(c->is_realized());

        // Verify execution (2*3 + 2*3 + 2*3 = 18)
        std::vector<float> host_verify(4);
        IAllocator* allocator = get_allocator(dev);
        allocator->copy_device_to_host(host_verify.data(), c->data->ptr, c->data->bytes);

        for (float val : host_verify) {
            EXPECT_FLOAT_EQ(val, 18.0f);
        }
    }
}

TEST_F(BackendRuntimeTest, SynchronizationBehavior) {
    EXPECT_NO_THROW(synchronize(DeviceEnum::CPU));

    #ifdef ORMI_USE_CUDA
    // If compiled with CUDA, verify execution if the hardware runner possesses a GPU
    bool cuda_hardware_present = false;
    for (auto dev : available_devices) {
        if (dev.type == DeviceEnum::CUDA) cuda_hardware_present = true;
    }
    if (cuda_hardware_present) {
        EXPECT_NO_THROW(synchronize(DeviceEnum::CUDA));
    }
    #else
        // If compiled WITHOUT CUDA, passing CUDA must throw a DeviceError
        EXPECT_THROW(synchronize(DeviceEnum::CUDA), ormi::DeviceError);
    #endif
}

TEST_F(BackendRuntimeTest, InvalidDeviceExecutionThrows) {
    Device invalid_dev(static_cast<DeviceEnum>(999));

    auto a = make_leaf({1}, 1.0f, Device(DeviceEnum::CPU));
    auto node = std::make_shared<Node>(ops::OpEnum::ADD, std::vector<std::shared_ptr<Node>>{a, a});
    node->device = invalid_dev;

    EXPECT_THROW(execute_op(node), ormi::DeviceError);
    EXPECT_THROW(synchronize(static_cast<DeviceEnum>(999)), ormi::DeviceError);
}