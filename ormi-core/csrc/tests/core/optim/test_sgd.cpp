#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/test_utils.h"

#include "core/allocator.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/optim/sgd.h"
#include "core/ops/pointwise/add.h"

using namespace ormi;
using namespace ormi::optim;

class SGDTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_node(
        const std::vector<int>& shape,
        float value,
        Device dev,
        bool requires_grad
    ) {
        auto buf = full(shape, value, dev, DType::Float32);
        return std::make_shared<Node>(buf, requires_grad);
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

        for (float value : host) {
            EXPECT_FLOAT_EQ(value, expected);
        }
    }
};

TEST_F(SGDTest, MathematicalStepExecution) {
    float initial_weight = 2.0f;
    float gradient_value = 0.5f;
    float learning_rate = 0.1f;

    float expected_weight = 1.95f;

    for (auto dev : available_devices) {
        auto p1 = make_node({2, 2}, initial_weight, dev, true);
        auto p2 = make_node({4}, initial_weight, dev, true);

        auto g1 = make_node({2, 2}, gradient_value, dev, false);
        auto g2 = make_node({4}, gradient_value, dev, false);

        SGD optimizer({
            {p1, g1},
            {p2, g2}
        }, learning_rate);

        optimizer.step();

        expect_all_value(p1, expected_weight);
        expect_all_value(p2, expected_weight);
    }
}

TEST_F(SGDTest, SkipMissingOrUnrealizedGradients) {
    float initial_weight = 3.14f;

    for (auto dev : available_devices) {
        auto p_no_grad = make_node({2}, initial_weight, dev, true);

        auto p_unrealized_grad = make_node({2}, initial_weight, dev, true);
        auto grad_input = make_node({2}, 1.0f, dev, false);
        auto unrealized_grad = ops::add(grad_input, grad_input);

        SGD optimizer({
            {p_no_grad, nullptr},
            {p_unrealized_grad, unrealized_grad}
        }, 0.1f);

        EXPECT_NO_THROW(optimizer.step());

        expect_all_value(p_no_grad, initial_weight);
        expect_all_value(p_unrealized_grad, initial_weight);
    }
}

TEST_F(SGDTest, UnsupportedDeviceThrowsDeviceError) {
    auto p = make_node({1}, 1.0f, Device::cpu(), true);
    auto g = make_node({1}, 1.0f, Device::cpu(), false);

    Device corrupted_device = Device::cpu();
    corrupted_device.type = static_cast<DeviceEnum>(999);

    p->device = corrupted_device;

    SGD optimizer({
        {p, g}
    }, 0.01f);

    EXPECT_THROW(optimizer.step(), ormi::DeviceError);
}