#include <gtest/gtest.h>

#include <utility>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/backend/id.h>

namespace {
    TEST(MemoryBufferTest, cpuBufferOwnsAllocatedMemory) {
        ormi::core::memory::Buffer buffer(ormi::core::Device::cpu(), 128);

        EXPECT_EQ(buffer.bytes(), 128u);
        EXPECT_NE(buffer.data(), nullptr);
        EXPECT_NE(buffer.allocator(), nullptr);
        EXPECT_TRUE(buffer.device().is_host());
    }

    TEST(MemoryBufferTest, zeroByteCpuBufferDoesNotAllocate) {
        ormi::core::memory::Buffer buffer(ormi::core::Device::cpu(), 0);

        EXPECT_EQ(buffer.bytes(), 0u);
        EXPECT_EQ(buffer.data(), nullptr);
        EXPECT_NE(buffer.allocator(), nullptr);
    }

    TEST(MemoryBufferTest, bufferIsMoveConstructible) {
        ormi::core::memory::Buffer source(ormi::core::Device::cpu(), 64);
        void* ptr = source.data();

        ormi::core::memory::Buffer target(std::move(source));

        EXPECT_EQ(target.bytes(), 64u);
        EXPECT_EQ(target.data(), ptr);
        EXPECT_EQ(source.bytes(), 0u);
        EXPECT_EQ(source.data(), nullptr);
    }

    TEST(MemoryBufferTest, bufferIsMoveAssignable) {
        ormi::core::memory::Buffer source(ormi::core::Device::cpu(), 64);
        ormi::core::memory::Buffer target(ormi::core::Device::cpu(), 32);
        void* ptr = source.data();

        target = std::move(source);

        EXPECT_EQ(target.bytes(), 64u);
        EXPECT_EQ(target.data(), ptr);
        EXPECT_EQ(source.bytes(), 0u);
        EXPECT_EQ(source.data(), nullptr);
    }

}
