#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "tests/test_utils.h"

#include "core/autograd.h"
#include "core/dtype.h"
#include "core/buffer.h"
#include "core/node.h"
#include "core/ops.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/reduce/sum.h"
#include "core/tapes/compute_tape.h"

#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/sub.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/reduce/sum.h"
#include "core/tapes/compute_tape.h"

using namespace ormi;

class AutogradTest : public ormi::testing::HardwareTestSuite {};

TEST_F(AutogradTest, GradReturnsInputGradients) {
    for (auto dev : available_devices) {
        auto buf_a = full({2, 2}, 2.0f, dev, DType::Float32);
        auto buf_b = full({2, 2}, 3.0f, dev, DType::Float32);

        auto a = std::make_shared<Node>(buf_a, true);
        auto b = std::make_shared<Node>(buf_b, true);

        auto c = std::make_shared<Node>(
            ops::OpEnum::ADD,
            std::vector<std::shared_ptr<Node>>{a, b}
        );

        auto grads = grad(c, {a, b});

        ASSERT_EQ(grads.size(), 2);

        ASSERT_NE(grads[0], nullptr);
        EXPECT_EQ(grads[0]->view.shape, (std::vector<int>{2, 2}));

        ASSERT_NE(grads[1], nullptr);
        EXPECT_EQ(grads[1]->view.shape, (std::vector<int>{2, 2}));
    }
}

TEST_F(AutogradTest, GradReturnsNullForUnusedInput) {
    for (auto dev : available_devices) {
        auto buf_a = full({2, 2}, 2.0f, dev, DType::Float32);
        auto buf_b = full({2, 2}, 3.0f, dev, DType::Float32);
        auto buf_unused = full({2, 2}, 4.0f, dev, DType::Float32);

        auto a = std::make_shared<Node>(buf_a, true);
        auto b = std::make_shared<Node>(buf_b, true);
        auto unused = std::make_shared<Node>(buf_unused, true);

        auto c = std::make_shared<Node>(
            ops::OpEnum::ADD,
            std::vector<std::shared_ptr<Node>>{a, b}
        );

        auto grads = grad(c, {a, unused});

        ASSERT_EQ(grads.size(), 2);

        ASSERT_NE(grads[0], nullptr);
        EXPECT_EQ(grads[0]->view.shape, (std::vector<int>{2, 2}));

        EXPECT_EQ(grads[1], nullptr);
    }
}

TEST_F(AutogradTest, SecondDerivativeOfCubic) {
    for (auto dev : available_devices) {
        auto x_buf = full({4}, 2.0f, dev, DType::Float32);
        auto x = std::make_shared<Node>(x_buf, true, "x");

        // y = sum(x * x * x)
        auto x2 = ops::mul(x, x);
        auto x3 = ops::mul(x2, x);
        auto y = ops::sum(x3);

        auto dy_dx = grad(y, {x})[0];

        // d/dx sum(3x^2) = 6x
        // x = 2 => 12
        auto d2y_dx2 = grad(dy_dx, {x})[0];

        ComputeTape tape;
        tape.capture({y, dy_dx, d2y_dx2});
        tape.execute();

        synchronize(dev.type);

        std::vector<float> h1(4);
        std::vector<float> h2(4);

        auto* allocator = get_allocator(dev);

        allocator->copy_device_to_host(
            h1.data(),
            dy_dx->data->ptr,
            dy_dx->data->bytes
        );

        allocator->copy_device_to_host(
            h2.data(),
            d2y_dx2->data->ptr,
            d2y_dx2->data->bytes
        );

        for (float v : h1) {
            EXPECT_NEAR(v, 12.0f, 1e-4f);
        }

        for (float v : h2) {
            EXPECT_NEAR(v, 12.0f, 1e-4f);
        }
    }
}

TEST_F(AutogradTest, GradWithExplicitSeedScalesGradient) {
    for (auto dev : available_devices) {
        auto x_buf = full({4}, 2.0f, dev, DType::Float32);
        auto x = std::make_shared<Node>(x_buf, true, "x");

        // y = x * x
        auto y = ops::mul(x, x);

        // seed = 3, so VJP = 3 * dy/dx = 3 * 2x = 12
        auto seed_buf = full({4}, 3.0f, dev, DType::Float32);
        auto seed = std::make_shared<Node>(seed_buf, false, "seed");

        auto dx = grad(y, {x}, seed)[0];

        ComputeTape tape;
        tape.capture({dx});
        tape.execute();

        synchronize(dev.type);

        std::vector<float> host(4);
        auto* allocator = get_allocator(dev);

        allocator->copy_device_to_host(
            host.data(),
            dx->data->ptr,
            dx->data->bytes
        );

        for (float v : host) {
            EXPECT_NEAR(v, 12.0f, 1e-4f);
        }
    }
}

TEST_F(AutogradTest, DiffComputesHigherOrderDerivative) {
    for (auto dev : available_devices) {
        auto x = std::make_shared<Node>(
            full({4}, 2.0f, dev, DType::Float32),
            true,
            "x"
        );

        // y = sum(x^3)
        auto x2 = ops::mul(x, x);
        auto x3 = ops::mul(x2, x);
        auto y = ops::sum(x3);

        auto y_x = diff(y, x, 1);
        auto y_xx = diff(y, x, 2);

        ComputeTape tape;
        tape.capture({y, y_x, y_xx});
        tape.execute();

        synchronize(dev.type);

        std::vector<float> h1(4);
        std::vector<float> h2(4);

        auto* allocator = get_allocator(dev);

        allocator->copy_device_to_host(
            h1.data(),
            y_x->data->ptr,
            y_x->data->bytes
        );

        allocator->copy_device_to_host(
            h2.data(),
            y_xx->data->ptr,
            y_xx->data->bytes
        );

        for (float v : h1) {
            EXPECT_NEAR(v, 12.0f, 1e-4f);
        }

        for (float v : h2) {
            EXPECT_NEAR(v, 12.0f, 1e-4f);
        }
    }
}

TEST_F(AutogradTest, PINNResidualHeatEquationStyle) {
    for (auto dev : available_devices) {
        auto x = std::make_shared<Node>(
            full({4}, 2.0f, dev, DType::Float32),
            true,
            "x"
        );

        auto t = std::make_shared<Node>(
            full({4}, 3.0f, dev, DType::Float32),
            true,
            "t"
        );

        // u(x,t) = x^2 + t^2
        auto x2 = ops::mul(x, x);
        auto t2 = ops::mul(t, t);
        auto u = ops::add(x2, t2);

        // u_t = 2t = 6
        auto u_t = diff(u, t, 1);

        // u_xx = 2
        auto u_xx = diff(u, x, 2);

        // alpha = 0.5
        auto alpha = scalar_like(x, 0.5f);

        // residual r = u_t - alpha * u_xx
        // r = 6 - 0.5 * 2 = 5
        auto residual = ops::sub(
            u_t,
            ops::mul(alpha, u_xx)
        );

        // loss = sum(r^2) = 4 elements * 25 = 100
        auto loss = ops::sum(
            ops::mul(residual, residual)
        );

        ComputeTape tape;
        tape.capture({u, u_t, u_xx, alpha, residual, loss});
        tape.execute();

        synchronize(dev.type);

        auto* allocator = get_allocator(dev);

        std::vector<float> host_residual(4);

        allocator->copy_device_to_host(
            host_residual.data(),
            residual->data->ptr,
            residual->data->bytes
        );

        for (float v : host_residual) {
            EXPECT_NEAR(v, 5.0f, 1e-4f);
        }

        float host_loss = 0.0f;

        allocator->copy_device_to_host(
            &host_loss,
            loss->data->ptr,
            loss->data->bytes
        );

        EXPECT_NEAR(host_loss, 100.0f, 1e-4f);
    }
}

TEST_F(AutogradTest, OnesLikeWorksForUnrealizedNode) {
    for (auto dev : available_devices) {
        auto x = std::make_shared<Node>(
            full({2, 3}, 2.0f, dev, DType::Float32),
            true,
            "x"
        );

        auto y = ops::mul(x, x); // unrealized internal node

        auto seed = ones_like(y, y->dtype, false);

        ASSERT_TRUE(seed);
        ASSERT_TRUE(seed->data);
        EXPECT_EQ(seed->view.shape, (std::vector<int>{2, 3}));
        EXPECT_EQ(seed->device, dev);
        EXPECT_EQ(seed->dtype, DType::Float32);

        ComputeTape tape;
        tape.capture({seed});
        tape.execute();

        synchronize(dev.type);

        std::vector<float> host(6);
        auto* allocator = get_allocator(dev);

        allocator->copy_device_to_host(
            host.data(),
            seed->data->ptr,
            seed->data->bytes
        );

        for (float v : host) {
            EXPECT_FLOAT_EQ(v, 1.0f);
        }
    }
}