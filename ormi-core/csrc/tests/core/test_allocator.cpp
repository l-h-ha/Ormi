#include <gtest/gtest.h>
#include <vector>

#include "tests/test_utils.h"
#include "core/allocator.h"
#include "core/dtype.h"

using namespace ormi;

class AllocatorTest : public ormi::testing::HardwareTestSuite {};

TEST_F(AllocatorTest, MemoryPoolRecyclesCorrectly) {
    for (auto dev : available_devices) {
        IAllocator* alloc = get_allocator(dev);
        
        size_t size_A = 1024;
        size_t size_B = 2048;

        // Allocate two blocks
        void* ptr1 = alloc->allocate(size_A);
        void* ptr2 = alloc->allocate(size_B);

        ASSERT_NE(ptr1, nullptr);
        ASSERT_NE(ptr2, nullptr);
        EXPECT_NE(ptr1, ptr2);

        // Free the first block
        alloc->free(ptr1, size_A);

        // Request a block of the SAME SIZE
        void* ptr3 = alloc->allocate(size_A);

        // The pool MUST return the exact same pointer to avoid reallocation overhead
        EXPECT_EQ(ptr1, ptr3) << "Memory pool failed to recycle pointer on device: " << static_cast<int>(dev);

        // Request a block of a DIFFERENT SIZE
        void* ptr4 = alloc->allocate(size_B);

        // Even though ptr1/ptr3 was accessed, size B should use a different pool bin
        EXPECT_NE(ptr3, ptr4);

        // Cleanup
        alloc->free(ptr3, size_A);
        alloc->free(ptr2, size_B);
        alloc->free(ptr4, size_B);
    }
}

TEST_F(AllocatorTest, FillAndHostTransfers) {
    for (auto dev : available_devices) {
        IAllocator* alloc = get_allocator(dev);
        size_t elements = 100;
        size_t bytes = elements * sizeof(float);

        void* d_ptr = alloc->allocate(bytes);

        // Test Float32 fill
        float fill_val = 3.14159f;
        alloc->fill_n(d_ptr, &fill_val, elements, DType::Float32);

        // Transfer to host
        std::vector<float> h_data(elements, 0.0f);
        alloc->copy_device_to_host(h_data.data(), d_ptr, bytes);

        for (size_t i = 0; i < elements; ++i) {
            EXPECT_FLOAT_EQ(h_data[i], fill_val);
        }

        // Test Host to Device explicitly
        h_data[0] = 99.9f;
        alloc->copy_host_to_device(d_ptr, h_data.data(), bytes);
        
        std::vector<float> h_verify(elements, 0.0f);
        alloc->copy_device_to_host(h_verify.data(), d_ptr, bytes);
        
        EXPECT_FLOAT_EQ(h_verify[0], 99.9f);
        EXPECT_FLOAT_EQ(h_verify[1], fill_val); // Rest should be unchanged

        alloc->free(d_ptr, bytes);
    }
}

TEST_F(AllocatorTest, DeviceToDeviceCopy) {
    for (auto dev : available_devices) {
        IAllocator* alloc = get_allocator(dev);
        size_t elements = 50;
        size_t bytes = elements * sizeof(float);

        void* src_ptr = alloc->allocate(bytes);
        void* dst_ptr = alloc->allocate(bytes);

        // Fill source with 42.0
        float val = 42.0f;
        alloc->fill_n(src_ptr, &val, elements, DType::Float32);

        // Copy source to destination on the device
        alloc->memcpy(dst_ptr, src_ptr, bytes);

        // Download destination to verify
        std::vector<float> h_data(elements, 0.0f);
        alloc->copy_device_to_host(h_data.data(), dst_ptr, bytes);

        for (size_t i = 0; i < elements; ++i) {
            EXPECT_FLOAT_EQ(h_data[i], 42.0f);
        }

        alloc->free(src_ptr, bytes);
        alloc->free(dst_ptr, bytes);
    }
}

TEST_F(AllocatorTest, HostVisibilityFlags) {
    for (auto dev : available_devices) {
        IAllocator* alloc = get_allocator(dev);
        
        if (dev == DeviceEnum::CPU) {
            EXPECT_TRUE(alloc->is_host_visible());
        } else if (dev == DeviceEnum::CUDA) {
            EXPECT_FALSE(alloc->is_host_visible());
        }
    }
}