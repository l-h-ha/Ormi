#include <gtest/gtest.h>
#include <vector>

#include "tests/test_utils.h"
#include "core/memory.h"
#include "core/allocator.h"

using namespace ormi;

class MemoryTest : public ormi::testing::HardwareTestSuite {};

TEST_F(MemoryTest, ZerosAndOnesAllocation) {
    std::vector<int> shape = {10, 10}; 
    size_t expected_bytes = 100 * sizeof(float);

    for (auto dev : available_devices) {
        auto zero_buf = zeros(shape, dev);
        auto one_buf = ones(shape, dev);

        ASSERT_NE(zero_buf, nullptr);
        EXPECT_EQ(zero_buf->bytes, expected_bytes);
        EXPECT_EQ(zero_buf->device.type, dev);
        
        std::vector<float> host_verify(100);
        IAllocator* allocator = get_allocator(dev);
        
        allocator->copy_device_to_host(host_verify.data(), zero_buf->ptr, expected_bytes);
        for (float val : host_verify) EXPECT_FLOAT_EQ(val, 0.0f);

        allocator->copy_device_to_host(host_verify.data(), one_buf->ptr, expected_bytes);
        for (float val : host_verify) EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

TEST_F(MemoryTest, EmptyAllocation) {
    std::vector<int> shape = {2, 3, 4}; // 24 elements
    size_t expected_bytes = 24 * sizeof(float);

    for (auto dev : available_devices) {
        auto empty_buf = empty(shape, dev);

        ASSERT_NE(empty_buf, nullptr);
        ASSERT_NE(empty_buf->ptr, nullptr);
        EXPECT_EQ(empty_buf->bytes, expected_bytes);
        EXPECT_EQ(empty_buf->shape, shape);
        EXPECT_EQ(empty_buf->device.type, dev);
    }
}

TEST_F(MemoryTest, FullAllocation) {
    std::vector<int> shape = {5, 5};
    size_t expected_elements = 25;
    float fill_value = 3.14159f;

    for (auto dev : available_devices) {
        auto buf = full(shape, fill_value, dev);

        std::vector<float> host_verify(expected_elements);
        IAllocator* allocator = get_allocator(dev);
        allocator->copy_device_to_host(host_verify.data(), buf->ptr, buf->bytes);

        for (float val : host_verify) EXPECT_FLOAT_EQ(val, fill_value);
    }
}

TEST_F(MemoryTest, ScalarAllocation) {
    float scalar_value = -42.0f;

    for (auto dev : available_devices) {
        auto buf = scalar(scalar_value, dev);

        EXPECT_EQ(buf->shape, std::vector<int>{1});
        EXPECT_EQ(buf->bytes, sizeof(float));

        std::vector<float> host_verify(1);
        IAllocator* allocator = get_allocator(dev);
        allocator->copy_device_to_host(host_verify.data(), buf->ptr, buf->bytes);

        EXPECT_FLOAT_EQ(host_verify[0], scalar_value);
    }
}

TEST_F(MemoryTest, LikeVariantsAllocation) {
    std::vector<int> shape = {3, 7};
    size_t expected_elements = 21;

    for (auto dev : available_devices) {
        auto base_buf = empty(shape, dev);
        auto z_like = zeros_like(base_buf);
        auto o_like = ones_like(base_buf);

        EXPECT_EQ(z_like->shape, shape);
        EXPECT_EQ(z_like->device.type, dev);
        EXPECT_EQ(o_like->shape, shape);
        EXPECT_EQ(o_like->device.type, dev);

        std::vector<float> host_verify(expected_elements);
        IAllocator* allocator = get_allocator(dev);

        allocator->copy_device_to_host(host_verify.data(), z_like->ptr, z_like->bytes);
        for (float val : host_verify) EXPECT_FLOAT_EQ(val, 0.0f);
        
        for (auto cross_dev : available_devices) {
            auto cross_like = zeros_like(base_buf, cross_dev);
            
            EXPECT_EQ(cross_like->shape, shape);
            EXPECT_EQ(cross_like->device.type, cross_dev);
            
            std::vector<float> cross_host(expected_elements);
            IAllocator* cross_allocator = get_allocator(cross_dev);
            cross_allocator->copy_device_to_host(cross_host.data(), cross_like->ptr, cross_like->bytes);
            
            for (float val : cross_host) EXPECT_FLOAT_EQ(val, 0.0f);
        }
    }
}

TEST_F(MemoryTest, ToStringFormatting) {
    for (auto dev : available_devices) {
        // Vector
        auto vec1d = full({3}, 2.5f, dev);
        std::string str1d = to_string(vec1d);
        EXPECT_EQ(str1d, "[2.5000, 2.5000, 2.5000]");

        // Matrix
        auto mat2d = full({2, 2}, 3.1415f, dev);
        std::string str2d = to_string(mat2d);
        std::string expected2d = "[[3.1415, 3.1415],\n [3.1415, 3.1415]]";
        EXPECT_EQ(str2d, expected2d);

        // Scalar
        auto scalar0d = empty({}, dev);
        float val = 42.0f;
        get_allocator(dev)->copy_host_to_device(scalar0d->ptr, &val, sizeof(float));
        std::string str0d = to_string(scalar0d);
        EXPECT_EQ(str0d, "42.0000");

        // Test operator
        std::ostringstream oss;
        oss << *vec1d;
        EXPECT_EQ(oss.str(), "[2.5000, 2.5000, 2.5000]");
    }
}

TEST_F(MemoryTest, OnesLikeValues) {
    std::vector<int> shape = {2, 2};
    size_t expected_elements = 4;

    for (auto dev : available_devices) {
        auto base_buf = empty(shape, dev);
        auto o_like = ones_like(base_buf);

        std::vector<float> host_verify(expected_elements);
        IAllocator* allocator = get_allocator(dev);

        allocator->copy_device_to_host(host_verify.data(), o_like->ptr, o_like->bytes);
        for (float val : host_verify) {
            EXPECT_FLOAT_EQ(val, 1.0f);
        }
    }
}