#include <array>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>

namespace {
    TEST(MemoryOpsTest, fillsCpuBufferBytes) {
        ormi::core::memory::Buffer buffer(ormi::core::Device::cpu(), 4);

        ormi::core::memory::fill(buffer, 0x7f);

        const auto* bytes = static_cast<const unsigned char*>(buffer.data());
        EXPECT_EQ(bytes[0], 0x7f);
        EXPECT_EQ(bytes[1], 0x7f);
        EXPECT_EQ(bytes[2], 0x7f);
        EXPECT_EQ(bytes[3], 0x7f);
    }

    TEST(MemoryOpsTest, zeroesCpuBufferBytes) {
        ormi::core::memory::Buffer buffer(ormi::core::Device::cpu(), 4);

        ormi::core::memory::fill(buffer, 0xff);
        ormi::core::memory::zero(buffer);

        const auto* bytes = static_cast<const unsigned char*>(buffer.data());
        EXPECT_EQ(bytes[0], 0x00);
        EXPECT_EQ(bytes[1], 0x00);
        EXPECT_EQ(bytes[2], 0x00);
        EXPECT_EQ(bytes[3], 0x00);
    }

    TEST(MemoryOpsTest, copiesCpuBufferBytes) {
        ormi::core::memory::Buffer src(ormi::core::Device::cpu(), 4);
        ormi::core::memory::Buffer dst(ormi::core::Device::cpu(), 4);

        auto* src_bytes = static_cast<unsigned char*>(src.data());
        src_bytes[0] = 1;
        src_bytes[1] = 2;
        src_bytes[2] = 3;
        src_bytes[3] = 4;

        ormi::core::memory::copy(dst, src);

        const auto* dst_bytes = static_cast<const unsigned char*>(dst.data());
        EXPECT_EQ(dst_bytes[0], 1);
        EXPECT_EQ(dst_bytes[1], 2);
        EXPECT_EQ(dst_bytes[2], 3);
        EXPECT_EQ(dst_bytes[3], 4);
    }

    TEST(MemoryOpsTest, rejectsOversizedBufferCopy) {
        ormi::core::memory::Buffer src(ormi::core::Device::cpu(), 4);
        ormi::core::memory::Buffer dst(ormi::core::Device::cpu(), 2);

        EXPECT_THROW(
            ormi::core::memory::copy(dst, src),
            ormi::MemoryError
        );
    }

    TEST(MemoryOpsTest, zeroBytePointerOperationsAreNoOps) {
        EXPECT_NO_THROW(
            ormi::core::memory::copy(
                nullptr,
                ormi::core::Device::cpu(),
                nullptr,
                ormi::core::Device::cpu(),
                0
            )
        );

        EXPECT_NO_THROW(
            ormi::core::memory::fill(nullptr, ormi::core::Device::cpu(), 0xff, 0)
        );
    }
}
