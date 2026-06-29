#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/test_utils.h"

#include "core/allocator.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/ops/pointwise/add.h"
#include "core/tapes/zero_grad_tape.h"

using namespace ormi;

class ZeroGradTapeTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_grad_node(
        const std::vector<int>& shape,
        float fill_val,
        Device dev
    ) {
        auto grad_buf = full(shape, fill_val, dev, DType::Float32);
        return std::make_shared<Node>(grad_buf, false);
    }

    void expect_all_zero(const std::shared_ptr<Node>& node) {
        ASSERT_TRUE(node);
        ASSERT_TRUE(node->data);

        std::vector<float> host(node->data->numel());
        auto* allocator = get_allocator(node->data->device);

        allocator->copy_device_to_host(
            host.data(),
            node->data->ptr,
            node->data->bytes
        );

        for (float val : host) {
            EXPECT_FLOAT_EQ(val, 0.0f);
        }
    }

    void expect_all_value(const std::shared_ptr<Node>& node, float expected) {
        ASSERT_TRUE(node);
        ASSERT_TRUE(node->data);

        std::vector<float> host(node->data->numel());
        auto* allocator = get_allocator(node->data->device);

        allocator->copy_device_to_host(
            host.data(),
            node->data->ptr,
            node->data->bytes
        );

        for (float val : host) {
            EXPECT_FLOAT_EQ(val, expected);
        }
    }
};

TEST_F(ZeroGradTapeTest, CaptureAndExecuteZerosExplicitGradNodes) {
    for (auto dev : available_devices) {
        auto a_grad = make_grad_node({2, 3}, 5.5f, dev);
        auto b_grad = make_grad_node({2, 3}, 10.0f, dev);

        ZeroGradTape tape;
        tape.capture({a_grad, b_grad});
        tape.execute();

        expect_all_zero(a_grad);
        expect_all_zero(b_grad);
    }
}

TEST_F(ZeroGradTapeTest, NullAndUnrealizedNodesAreIgnored) {
    for (auto dev : available_devices) {
        auto a = std::make_shared<Node>(
            full({4}, 1.0f, dev, DType::Float32),
            false
        );

        auto b = std::make_shared<Node>(
            full({4}, 2.0f, dev, DType::Float32),
            false
        );

        auto unrealized = ops::add(a, b);

        ZeroGradTape tape;

        EXPECT_NO_THROW(tape.capture({nullptr, unrealized}));
        EXPECT_NO_THROW(tape.execute());
    }
}

TEST_F(ZeroGradTapeTest, DoesNotZeroNodesThatWereNotCaptured) {
    for (auto dev : available_devices) {
        auto captured_grad = make_grad_node({2, 3}, 5.5f, dev);
        auto untouched_grad = make_grad_node({2, 3}, 7.0f, dev);

        ZeroGradTape tape;
        tape.capture({captured_grad});
        tape.execute();

        expect_all_zero(captured_grad);
        expect_all_value(untouched_grad, 7.0f);
    }
}