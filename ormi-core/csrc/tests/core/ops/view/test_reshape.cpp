#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "tests/test_utils.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "core/registry.h"
#include "core/attributes.h"
#include "core/ops/view/reshape.h"
#include "core/ops/view/transpose.h"

using namespace ormi;

class OpsReshapeTest : public ormi::testing::HardwareTestSuite {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape, bool requires_grad, Device dev) {
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, requires_grad, dev);
    }
};

TEST_F(OpsReshapeTest, ContiguousReshapeView) {
    for (auto dev : available_devices) {
        // Base shape [2, 3, 4], Elements = 24. Contiguous strides [12, 4, 1]
        auto a = make_leaf({2, 3, 4}, false, dev);
        
        // Split [4] into [2, 2] -> New shape [2, 3, 2, 2]
        auto b = ops::reshape(a, {2, 3, 2, 2});
        EXPECT_EQ(b->view.shape, (std::vector<int>{2, 3, 2, 2}));
        EXPECT_EQ(b->view.strides, (std::vector<int>{12, 4, 2, 1}));

        // Merge [2, 3] into [6] -> New shape [6, 4]
        auto c = ops::reshape(a, {6, 4});
        EXPECT_EQ(c->view.shape, (std::vector<int>{6, 4}));
        EXPECT_EQ(c->view.strides, (std::vector<int>{4, 1}));
    }
}

TEST_F(OpsReshapeTest, NonContiguousReshapeThrows) {
    for (auto dev : available_devices) {
        // Base shape [2, 3, 4]. Contiguous strides [12, 4, 1]
        auto a = make_leaf({2, 3, 4}, false, dev);
        
        // Transpose swaps dimensions. Shape -> [2, 4, 3], Strides -> [12, 1, 4]
        // This makes the memory non-contiguous.
        std::vector<int> axes = {0, 2, 1};
        auto transposed = ops::transpose(a, axes);

        // Attempting to merge the transposed dimensions [4, 3] into [12] 
        // requires jumping back and forth in memory. 
        // A zero-copy view is impossible, so an exception must be thrown.
        EXPECT_THROW(ops::reshape(transposed, {2, 12}), std::runtime_error);
    }
}

TEST_F(OpsReshapeTest, AdjointGraphConstruction) {
    for (auto dev : available_devices) {
        auto a = make_leaf({2, 3, 4}, true, dev);
        auto z = ops::reshape(a, {6, 4});

        // Seed gradient
        z->grad = make_leaf({6, 4}, false, dev);

        // Adjoint
        auto adjoint_func = Registry<AdjointFunc>::get(ops::OpEnum::RESHAPE);
        ASSERT_NE(adjoint_func, nullptr);
        adjoint_func(z);

        // Gradient of reshape is just a reshape back to the input dimensions
        ASSERT_NE(a->grad, nullptr);
        EXPECT_EQ(a->grad->op, ops::OpEnum::RESHAPE);
        EXPECT_EQ(a->grad->view.shape, (std::vector<int>{2, 3, 4}));

        // Validate the target attributes
        const auto& res_attrs = std::get<ReshapeAttrs>(a->grad->attrs);
        EXPECT_EQ(res_attrs.shape, (std::vector<int>{2, 3, 4}));
    }
}