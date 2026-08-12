#include <cstddef>

#include <gtest/gtest.h>

#include "platform/platform.h"

namespace {
    TEST(PlatformCPUTest, reportsAtLeastOneCore) {
        EXPECT_GE(ormi::platform::cpu_cores(), std::size_t{1});
    }

    TEST(PlatformCPUTest, reportsConsistentVectorCapabilities) {
        if (ormi::platform::supports_fma()) {
            EXPECT_TRUE(ormi::platform::supports_avx2());
        }
        if (ormi::platform::supports_f16c()) {
            EXPECT_TRUE(ormi::platform::supports_avx2());
        }
    }
}
