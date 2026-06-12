#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "backend/cuda/compiler.cuh"
#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/sin.h"
#include "core/ops/blas/matmul.h"

using namespace ormi;

class CUDACompilerTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape) {
        auto dev = Device(DeviceEnum::CUDA);
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, false, dev);
    }
    
    void SetUp() override {
        HardwareTestSuite::SetUp();
        bool has_cuda = false;
        for (auto dev : available_devices) {
            if (dev == DeviceEnum::CUDA) has_cuda = true;
        }
        if (!has_cuda) GTEST_SKIP() << "Skipping CUDA compiler tests: CUDA device unavailable.";
    }
};

TEST_F(CUDACompilerTest, BasicKernelStringGeneration) {
    auto a = make_leaf({2, 2});
    auto b = make_leaf({2, 2});
    auto c = ops::add(a, b);

    auto result = cuda::compile(c);

    EXPECT_EQ(result.leaf_nodes.size(), 2);
    EXPECT_EQ(result.leaf_names.size(), 2);
    EXPECT_EQ(result.tensor_size, 4);

    // Verify signature and bounds check
    EXPECT_NE(result.kernel_code.find("extern \"C\" __global__"), std::string::npos);
    EXPECT_NE(result.kernel_code.find("void fused_kernel(float* OUT_Z"), std::string::npos);
    EXPECT_NE(result.kernel_code.find("if (idx >= 4) return;"), std::string::npos);

    // Verify operation mapping
    EXPECT_NE(result.kernel_code.find("+"), std::string::npos);
}

TEST_F(CUDACompilerTest, NodeDeduplication) {
    auto a = make_leaf({2, 2});
    auto c = ops::add(a, a); 

    auto result = cuda::compile(c);

    // 1 leaf node mapped to the kernel despite being used twice
    EXPECT_EQ(result.leaf_nodes.size(), 1);
    
    // Check that the same variable name is used for both operands
    std::string expected_op = "val_" + result.leaf_names[0] + " + val_" + result.leaf_names[0];
    EXPECT_NE(result.kernel_code.find(expected_op), std::string::npos);
}

TEST_F(CUDACompilerTest, MathFunctionMapping) {
    auto a = make_leaf({2, 2});
    auto b = ops::sin(a);

    auto result = cuda::compile(b);

    // Verify correct C-math function mapping
    EXPECT_NE(result.kernel_code.find("sinf("), std::string::npos);
}

TEST_F(CUDACompilerTest, GraphBreakerValidation) {
    auto a = make_leaf({2, 2});
    auto b = make_leaf({2, 2});
    auto c = ops::matmul(a, b);

    // The compiler must refuse to fuse a graph breaking operation
    EXPECT_THROW(cuda::compile(c), std::runtime_error);
}