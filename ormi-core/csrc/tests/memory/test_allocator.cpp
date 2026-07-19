#include <cstdint>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/runtime/dispatch/device.h>

namespace {
    TEST(MemoryAllocatorTest, cpuAllocatorIsHostVisible) {
        auto* allocator = ormi::core::runtime::device::allocator(ormi::core::Device::cpu());

        ASSERT_NE(allocator, nullptr);
        EXPECT_TRUE(allocator->host_visible());
        EXPECT_NO_THROW(allocator->empty_cache());
    }

    TEST(MemoryAllocatorTest, cpuAllocatorReturnsAlignedMemory) {
        auto* allocator = ormi::core::runtime::device::allocator(ormi::core::Device::cpu());

        void* ptr = allocator->allocate(1);

        ASSERT_NE(ptr, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0u);

        allocator->deallocate(ptr, 1);
    }

    TEST(MemoryAllocatorTest, cpuAllocatorAcceptsZeroByteAllocation) {
        auto* allocator = ormi::core::runtime::device::allocator(ormi::core::Device::cpu());

        EXPECT_EQ(allocator->allocate(0), nullptr);
        EXPECT_NO_THROW(allocator->deallocate(nullptr, 0));
    }

    TEST(MemoryAllocatorTest, cpuAllocatorReusesCachedBlockWithSameSizeClass) {
        auto* allocator = ormi::core::runtime::device::allocator(ormi::core::Device::cpu());

        void* first = allocator->allocate(33);
        allocator->deallocate(first, 33);

        void* second = allocator->allocate(33);
        allocator->deallocate(second, 33);

        EXPECT_EQ(second, first);
        allocator->empty_cache();
    }

    TEST(MemoryAllocatorTest, cpuAllocatorKeepsSizeClassesSeparate) {
        auto* allocator = ormi::core::runtime::device::allocator(ormi::core::Device::cpu());

        void* small = allocator->allocate(64);
        void* large = allocator->allocate(128);

        allocator->deallocate(small, 64);
        allocator->deallocate(large, 128);

        void* next_large = allocator->allocate(128);
        void* next_small = allocator->allocate(64);

        EXPECT_EQ(next_large, large);
        EXPECT_EQ(next_small, small);

        allocator->deallocate(next_large, 128);
        allocator->deallocate(next_small, 64);
        allocator->empty_cache();
    }
}
