#include <gtest/gtest.h>

#include "core/runtime/backends.h"
#include "backends/cpu/backend.h"

namespace {
    TEST(CPUBackendTest, createsRequiredFunctionTable) {
        const auto backend = ormi::backends::cpu::init();

        EXPECT_EQ(backend.abi, ormi::core::runtime::backend_abi);
        EXPECT_NE(backend.memory.allocator, nullptr);
        EXPECT_NE(backend.memory.synchronize, nullptr);
        EXPECT_NE(backend.memory.copy, nullptr);
        EXPECT_EQ(backend.memory.direct_copy, nullptr);
        EXPECT_NE(backend.memory.fill, nullptr);
        EXPECT_NE(backend.memory.fill_scalar, nullptr);
        EXPECT_NE(backend.ops.cast, nullptr);
        EXPECT_NE(backend.ops.sgd, nullptr);
        EXPECT_NE(backend.ops.matmul, nullptr);
        EXPECT_NE(backend.ops.reduce, nullptr);
        EXPECT_NE(backend.ops.binary, nullptr);
        EXPECT_NE(backend.ops.unary, nullptr);
        EXPECT_NE(backend.fractional.execute, nullptr);
        EXPECT_EQ(backend.fractional.prepare, nullptr);
        EXPECT_NE(backend.fusion.supports, nullptr);
        EXPECT_NE(backend.fusion.prepare, nullptr);
        EXPECT_NE(backend.fusion.execute, nullptr);
        EXPECT_EQ(backend.recording.record, nullptr);
        EXPECT_EQ(backend.recording.replay, nullptr);
        EXPECT_EQ(backend.recording.release, nullptr);
    }

    TEST(CPUBackendTest, staticBackendsInstallsCPUBackend) {
        auto& backends = ormi::core::runtime::Backends::instance();
        const auto cpu = backends.find("cpu");
        ASSERT_NE(cpu, ormi::core::runtime::backend_invalid);

        const auto& backend = backends.require(cpu);
        EXPECT_EQ(backend.abi, ormi::core::runtime::backend_abi);
    }
}
