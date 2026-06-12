#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

#include "tests/test_utils.h"
#include "core/ops.h"
#include "core/node.h"
#include "core/memory.h"
#include "core/allocator.h"
#include "core/ops/blas/matmul.h"
#include "core/ops/reduce/sum.h"

using namespace ormi;

class OpsTest : public ormi::testing::HardwareTestSuite {
protected:
    std::vector<float> download_buffer(const std::shared_ptr<Buffer>& buf) {
        std::vector<float> host_data(buf->bytes / sizeof(float));
        IAllocator* allocator = get_allocator(buf->device.type);
        allocator->copy_device_to_host(host_data.data(), buf->ptr, buf->bytes);
        return host_data;
    }
};

TEST_F(OpsTest, MatMulStandard2D) {
    for (auto dev : available_devices) {
        auto bufA = full({2, 3}, 2.0f, dev);
        auto bufB = full({3, 2}, 3.0f, dev);

        auto nodeA = std::make_shared<Node>(bufA, false, dev);
        auto nodeB = std::make_shared<Node>(bufB, false, dev);
        
        auto result_node = ops::matmul(nodeA, nodeB);

        ASSERT_NE(result_node, nullptr);
        EXPECT_FALSE(result_node->is_realized());
        EXPECT_EQ(result_node->view.shape, (std::vector<int>{2, 2}));
        
        result_node->realize();
        auto host_data = download_buffer(result_node->data);
        
        for (float val : host_data) {
            EXPECT_FLOAT_EQ(val, 18.0f) << "Standard MatMul failed on device: " << static_cast<int>(dev);
        }
    }
}

TEST_F(OpsTest, MatMulBatchedBroadcasting) {
    for (auto dev : available_devices) {
        // Shape A: [2, 1, 2, 3] 
        // Shape B: [1, 2, 3, 2] 
        // Output Shape: [2, 2, 2, 2]
        auto bufA = full({2, 1, 2, 3}, 1.5f, dev);
        auto bufB = full({1, 2, 3, 2}, 2.0f, dev);

        auto nodeA = std::make_shared<Node>(bufA, false, dev);
        auto nodeB = std::make_shared<Node>(bufB, false, dev);
        
        auto result_node = ops::matmul(nodeA, nodeB);
        EXPECT_EQ(result_node->view.shape, (std::vector<int>{2, 2, 2, 2}));

        result_node->realize();
        auto host_data = download_buffer(result_node->data);
        
        size_t expected_elements = 2 * 2 * 2 * 2;
        ASSERT_EQ(host_data.size(), expected_elements);
        for (size_t i = 0; i < expected_elements; ++i) {
            EXPECT_FLOAT_EQ(host_data[i], 9.0f) << "Batched MatMul failed at index " << i;
        }
    }
}

TEST_F(OpsTest, MatMulEdgeCasesThrows) {
    for (auto dev : available_devices) {
        // Inner dimension mismatch
        auto bufA_inner = full({2, 3}, 1.0f, dev);
        auto bufB_inner = full({2, 2}, 1.0f, dev); 
        auto nodeA_inner = std::make_shared<Node>(bufA_inner, false, dev);
        auto nodeB_inner = std::make_shared<Node>(bufB_inner, false, dev);
        EXPECT_THROW(ops::matmul(nodeA_inner, nodeB_inner), std::runtime_error);

        // Too few dimensions (1D tensor)
        auto bufA_1d = full({3}, 1.0f, dev);
        auto bufB_1d = full({3}, 1.0f, dev); 
        auto nodeA_1d = std::make_shared<Node>(bufA_1d, false, dev);
        auto nodeB_1d = std::make_shared<Node>(bufB_1d, false, dev);
        EXPECT_THROW(ops::matmul(nodeA_1d, nodeB_1d), std::runtime_error);

        // Unbroadcastable batch dimensions
        auto bufA_batch = full({2, 2, 3}, 1.0f, dev);
        auto bufB_batch = full({3, 3, 2}, 1.0f, dev); 
        auto nodeA_batch = std::make_shared<Node>(bufA_batch, false, dev);
        auto nodeB_batch = std::make_shared<Node>(bufB_batch, false, dev);
        EXPECT_THROW(ops::matmul(nodeA_batch, nodeB_batch), std::runtime_error);
    }
}

TEST_F(OpsTest, SumGlobalReduction) {
    for (DeviceEnum dev : available_devices) {
        auto bufA = full({2, 3}, 2.0f, dev);
        auto nodeA = std::make_shared<Node>(bufA, false, dev);

        // Test without keepdims (Output: Scalar)
        auto result_flat = ops::sum(nodeA, {}, false);
        EXPECT_EQ(result_flat->view.shape, (std::vector<int>{})); 
        result_flat->realize();
        auto data_flat = download_buffer(result_flat->data);
        EXPECT_FLOAT_EQ(data_flat[0], 12.0f);

        // Test with keepdims (Output: [1, 1])
        auto result_kept = ops::sum(nodeA, {}, true);
        EXPECT_EQ(result_kept->view.shape, (std::vector<int>{1, 1})); 
        result_kept->realize();
        auto data_kept = download_buffer(result_kept->data);
        EXPECT_FLOAT_EQ(data_kept[0], 12.0f);
    }
}

TEST_F(OpsTest, SumAxisSpecificReduction) {
    for (DeviceEnum dev : available_devices) {
        std::vector<float> init_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        auto bufA = empty({2, 3}, dev);
        get_allocator(dev)->copy_host_to_device(bufA->ptr, init_data.data(), bufA->bytes);
        auto nodeA = std::make_shared<Node>(bufA, false, dev);

        // Axis 1 without keepdims -> Shape [2]
        auto res_axis1_flat = ops::sum(nodeA, {1}, false);
        EXPECT_EQ(res_axis1_flat->view.shape, (std::vector<int>{2}));
        res_axis1_flat->realize();
        auto data_axis1_flat = download_buffer(res_axis1_flat->data);
        EXPECT_FLOAT_EQ(data_axis1_flat[0], 6.0f);
        EXPECT_FLOAT_EQ(data_axis1_flat[1], 15.0f);

        // Axis 1 with keepdims -> Shape [2, 1]
        auto res_axis1_kept = ops::sum(nodeA, {1}, true);
        EXPECT_EQ(res_axis1_kept->view.shape, (std::vector<int>{2, 1}));
        res_axis1_kept->realize();
        auto data_axis1_kept = download_buffer(res_axis1_kept->data);
        EXPECT_FLOAT_EQ(data_axis1_kept[0], 6.0f);
        EXPECT_FLOAT_EQ(data_axis1_kept[1], 15.0f);
    }
}

TEST_F(OpsTest, SumMultiAxisReduction) {
    for (DeviceEnum dev : available_devices) {
        auto bufA = full({2, 3, 2}, 1.0f, dev);
        auto nodeA = std::make_shared<Node>(bufA, false, dev);

        auto res = ops::sum(nodeA, {0, 2}, false);
        EXPECT_EQ(res->view.shape, (std::vector<int>{3}));
        
        res->realize();
        auto data = download_buffer(res->data);
        
        ASSERT_EQ(data.size(), 3);
        EXPECT_FLOAT_EQ(data[0], 4.0f);
        EXPECT_FLOAT_EQ(data[1], 4.0f);
        EXPECT_FLOAT_EQ(data[2], 4.0f);
    }
}

TEST_F(OpsTest, SafetyDeviceMismatchThrows) {    
    if (available_devices.size() >= 2) {
        auto dev_cpu = available_devices[0]; 
        auto dev_cuda = available_devices[1];

        auto bufA = full({2, 2}, 1.0f, dev_cpu);
        auto bufB = full({2, 2}, 1.0f, dev_cuda);

        auto nodeA = std::make_shared<Node>(bufA, false, dev_cpu);
        auto nodeB = std::make_shared<Node>(bufB, false, dev_cuda);

        EXPECT_THROW(ops::matmul(nodeA, nodeB), std::runtime_error);
    }
}