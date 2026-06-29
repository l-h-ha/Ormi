#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/buffer.h"
#include "core/allocator.h"
#include "core/ops.h"
#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/mul.h"
#include "core/tapes/compute_tape.h"
#include "backend/runtime.h"

using namespace ormi;

class ComputeTapeTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, float fill_val, Device dev) {
        auto buf = full(shape, fill_val, dev, DType::Float32);
        return std::make_shared<Node>(buf, false);
    }
};

TEST_F(ComputeTapeTest, CaptureFiltersLeavesButDoesNotExecuteImmediately) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 2}, 1.5f, dev);
        auto b = make_leaf({2, 2}, 2.5f, dev);

        auto c = ops::add(a, b);

        EXPECT_FALSE(c->is_realized());

        ComputeTape tape;

        tape.capture({c});

        // Capture only records topology now.
        EXPECT_FALSE(c->is_realized());

        tape.execute();

        EXPECT_TRUE(c->is_realized());

        std::vector<float> host_verify(4);
        IAllocator* allocator = get_allocator(dev);

        allocator->copy_device_to_host(
            host_verify.data(),
            c->data->ptr,
            c->data->bytes
        );

        for (float val : host_verify) {
            EXPECT_FLOAT_EQ(val, 4.0f);
        }
    }
}

TEST_F(ComputeTapeTest, ExecuteReplaysCachedExecutionOrder) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 2}, 2.0f, dev);
        auto b = make_leaf({2, 2}, 3.0f, dev);

        auto c = ops::add(a, b); // 2 + 3 = 5

        ComputeTape tape;
        tape.capture({c});

        // capture should only record topology now
        EXPECT_FALSE(c->is_realized());

        tape.execute();

        EXPECT_TRUE(c->is_realized());

        std::vector<float> host_verify(4);
        auto* allocator = get_allocator(dev);

        allocator->copy_device_to_host(
            host_verify.data(),
            c->data->ptr,
            c->data->bytes
        );

        for (float val : host_verify) {
            EXPECT_FLOAT_EQ(val, 5.0f);
        }

        // Replay again from cached execution order.
        tape.execute();

        allocator->copy_device_to_host(
            host_verify.data(),
            c->data->ptr,
            c->data->bytes
        );

        for (float val : host_verify) {
            EXPECT_FLOAT_EQ(val, 5.0f);
        }
    }
}

TEST_F(ComputeTapeTest, LeafOnlyGraphDoesNothing) {
    for (auto dev : available_devices) {
        auto a = make_leaf({4}, 1.0f, dev);
        
        ComputeTape tape;
        
        // Capturing a pure leaf graph should bypass execution_order caching completely
        EXPECT_NO_THROW(tape.capture({a}));
        
        // Executing an empty pipeline must behave safely as a no-op
        EXPECT_NO_THROW(tape.execute());
    }
}

TEST_F(ComputeTapeTest, DeepGraphRelativeOrdering) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 2}, 2.0f, dev);
        auto b = make_leaf({2, 2}, 3.0f, dev);
        auto c = make_leaf({2, 2}, 4.0f, dev);

        // Graph path: D = (A + B) * C
        auto a_plus_b = ops::add(a, b);
        auto d = ops::mul(a_plus_b, c);

        ComputeTape tape;
        tape.capture({d});

        // Capture records only; it must not execute.
        EXPECT_FALSE(a_plus_b->is_realized());
        EXPECT_FALSE(d->is_realized());

        tape.execute();

        EXPECT_TRUE(a_plus_b->is_realized());
        EXPECT_TRUE(d->is_realized());

        // Verify ordered replay dependency tracking.
        a_plus_b->data = nullptr;
        d->data = nullptr;

        tape.execute();

        std::vector<float> host_verify(4);
        IAllocator* allocator = get_allocator(dev);

        allocator->copy_device_to_host(
            host_verify.data(),
            d->data->ptr,
            d->data->bytes
        );

        for (float val : host_verify) {
            EXPECT_FLOAT_EQ(val, 20.0f);
        }
    }
}