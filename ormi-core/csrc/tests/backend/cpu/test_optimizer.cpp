#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "core/buffer.h"
#include "core/dtype.h"
#include "backend/cpu/optimizer.h"

using namespace ormi;

class CPUOptimizerTest : public ::testing::Test {
protected:
    std::shared_ptr<Buffer> make_buffer(int size, float fill_val) {
        return full({size}, fill_val, Device::cpu(), DType::Float32);
    }

    void expect_buffer_values(const std::shared_ptr<Buffer>& buf, float expected_val) {
        int elements = buf->bytes / sizeof(float);
        const float* ptr = static_cast<const float*>(buf->ptr);
        for (int i = 0; i < elements; ++i) {
            EXPECT_FLOAT_EQ(ptr[i], expected_val);
        }
    }
};

TEST_F(CPUOptimizerTest, SGDBasicStepMath) {
    int size = 10;
    float initial_weight = 2.0f;
    float gradient = 0.5f;
    float lr = 0.1f;

    auto weight = make_buffer(size, initial_weight);
    auto grad = make_buffer(size, gradient);
    cpu::sgd_step(grad, weight, lr);

    // 2.0 - (0.5 * 0.1) = 1.95
    expect_buffer_values(weight, 1.95f);
}

TEST_F(CPUOptimizerTest, SGDZeroLearningRate) {
    int size = 5;
    float initial_weight = 3.14f;
    float gradient = 100.0f; // Large gradient to prove LR scales it to zero
    float lr = 0.0f;

    auto weight = make_buffer(size, initial_weight);
    auto grad = make_buffer(size, gradient);

    cpu::sgd_step(grad, weight, lr);

    // Weights remain completely unchanged
    expect_buffer_values(weight, initial_weight);
}

TEST_F(CPUOptimizerTest, SGDZeroGradient) {
    int size = 5;
    float initial_weight = -1.5f;
    float gradient = 0.0f;
    float lr = 0.01f;

    auto weight = make_buffer(size, initial_weight);
    auto grad = make_buffer(size, gradient);

    cpu::sgd_step(grad, weight, lr);

    // Weights remain completely unchanged
    expect_buffer_values(weight, initial_weight);
}

TEST_F(CPUOptimizerTest, SGDAggressiveLearningRate) {
    int size = 4;
    float initial_weight = 10.0f;
    float gradient = 2.0f;
    float lr = 10.0f; // Unusually high LR to check scaling boundaries

    auto weight = make_buffer(size, initial_weight);
    auto grad = make_buffer(size, gradient);

    cpu::sgd_step(grad, weight, lr);

    // 10.0 - (2.0 * 10.0) = -10.0
    expect_buffer_values(weight, -10.0f);
}