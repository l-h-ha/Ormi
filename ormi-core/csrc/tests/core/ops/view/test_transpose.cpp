#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <stdexcept>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/attributes.h"
#include "core/ops/view/transpose.h"

using namespace ormi;

class OpsTransposeTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsTransposeTest, ZeroCopyStrideInference) {
    for (auto dev : available_devices) {
        // Base shape: [2, 3, 4], Contiguous strides: [12, 4, 1]
        auto a = make_leaf({2, 3, 4}, false, dev);
        
        // Permute to [4, 2, 3] using axes (2, 0, 1)
        std::vector<int> axes = {2, 0, 1};
        auto b = ops::transpose(a, axes);

        EXPECT_EQ(b->op, ops::OpEnum::TRANSPOSE);
        
        // Shape should be re-ordered based on axes
        EXPECT_EQ(b->view.shape, (std::vector<int>{4, 2, 3}));
        
        // Strides MUST be reordered to match the same permutation
        // Original: [12, 4, 1] -> Permuted (2, 0, 1): [1, 12, 4]
        EXPECT_EQ(b->view.strides, (std::vector<int>{1, 12, 4}));
    }
}

TEST_F(OpsTransposeTest, InvalidAxesThrow) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, false, dev);
        
        // Attempting to pass too few or too many axes must throw
        std::vector<int> too_few = {1, 0};
        EXPECT_THROW(ops::transpose(a, too_few), std::runtime_error);

        std::vector<int> too_many = {0, 1, 2, 3};
        EXPECT_THROW(ops::transpose(a, too_many), std::runtime_error);
    }
}

TEST_F(OpsTransposeTest, AdjointInversePermutation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);
        
        // Forward permutation: {2, 0, 1}
        std::vector<int> axes = {2, 0, 1}; 
        auto z = ops::transpose(a, axes);

        // Seed output gradient with the transposed shape [4, 2, 3]
        z->grad = make_leaf({4, 2, 3}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::TRANSPOSE);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(z);

        // Gradient node should be another TRANSPOSE
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::TRANSPOSE);
        
        // Shape must return to original [2, 3, 4]
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3, 4}));

        // Forward: {2, 0, 1}. The inverse that maps it back is {1, 2, 0}
        const auto& inv_attrs = std::get<PermuteAttrs>(a->grad->attrs);
        EXPECT_EQ(inv_attrs.axes, (std::vector<int>{1, 2, 0}));
    }
}

///
/// 4. Adjoint Gradient Accumulation
///

TEST_F(OpsTransposeTest, AdjointGradientAccumulation) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3}, true, dev);
        auto z = ops::transpose(a, {1, 0});
        
        z->grad = make_leaf({3, 2}, false, dev);

        // Pre-populate input grad to simulate accumulated gradients
        a->grad = make_leaf({2, 3}, false, dev);

        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::TRANSPOSE);
        adjoint_func(z);

        // The top node must be an ADD operation merging old grad with transposed grad
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::ADD);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3}));
    }
}