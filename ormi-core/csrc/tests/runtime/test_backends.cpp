#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/core/runtime/backend_id.h>
#include <ormi/core/runtime/backends.h>

namespace {
    TEST(BackendsTest, staticInstanceInitializesCPUBackend) {
        auto& backends = ormi::core::runtime::Backends::instance();

        EXPECT_TRUE(backends.available(ormi::core::runtime::backend_id_cpu));

        const auto& cpu = backends.require(ormi::core::runtime::backend_id_cpu);
        EXPECT_EQ(cpu.abi, ormi::core::runtime::backend_abi);
        EXPECT_EQ(cpu.id, ormi::core::runtime::backend_id_cpu);
        EXPECT_EQ(cpu.name, "cpu");
        EXPECT_NE(cpu.allocator, nullptr);
        EXPECT_NE(cpu.synchronize, nullptr);
        EXPECT_NE(cpu.copy, nullptr);
        EXPECT_NE(cpu.fill, nullptr);
        EXPECT_NE(cpu.fill_scalar, nullptr);
        EXPECT_NE(cpu.cast, nullptr);
        EXPECT_NE(cpu.sgd, nullptr);
        EXPECT_NE(cpu.matmul, nullptr);
        EXPECT_NE(cpu.reduce, nullptr);
        EXPECT_NE(cpu.all_finite, nullptr);
        EXPECT_NE(cpu.pointwise_binary, nullptr);
        EXPECT_NE(cpu.pointwise_unary, nullptr);
    }

    TEST(BackendsTest, resolvesBackendNames) {
        auto& backends = ormi::core::runtime::Backends::instance();

        EXPECT_EQ(backends.find("cpu"), ormi::core::runtime::backend_id_cpu);
        EXPECT_EQ(backends.find("cuda"), ormi::core::runtime::backend_id_cuda);
        EXPECT_EQ(backends.find(""), ormi::core::runtime::backend_invalid);
        EXPECT_EQ(backends.find("CPU"), ormi::core::runtime::backend_invalid);
        EXPECT_EQ(backends.find("metal"), ormi::core::runtime::backend_invalid);
    }
}
