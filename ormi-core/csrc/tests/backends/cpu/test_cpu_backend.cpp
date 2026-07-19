#include <gtest/gtest.h>

#include <ormi/core/runtime/backend_id.h>
#include <ormi/core/runtime/backends.h>

#include "backends/cpu/backend.h"

namespace {
    TEST(CPUBackendTest, createsRequiredFunctionTable) {
        const auto backend = ormi::backends::cpu::init();

        EXPECT_EQ(backend.id, ormi::core::runtime::backend_id_cpu);
        EXPECT_EQ(backend.name, "cpu");
        EXPECT_NE(backend.allocator, nullptr);
        EXPECT_NE(backend.synchronize, nullptr);
        EXPECT_NE(backend.copy, nullptr);
        EXPECT_NE(backend.fill, nullptr);
        EXPECT_NE(backend.fill_scalar, nullptr);
        EXPECT_NE(backend.cast, nullptr);
        EXPECT_NE(backend.sgd, nullptr);
        EXPECT_NE(backend.matmul, nullptr);
        EXPECT_NE(backend.reduce, nullptr);
        EXPECT_NE(backend.pointwise_binary, nullptr);
        EXPECT_NE(backend.pointwise_unary, nullptr);
        EXPECT_NE(backend.fusion, nullptr);
    }

    TEST(CPUBackendTest, staticBackendsInstallsCPUBackend) {
        const auto& backend =
            ormi::core::runtime::Backends::instance().require(
                ormi::core::runtime::backend_id_cpu
            );

        EXPECT_EQ(backend.id, ormi::core::runtime::backend_id_cpu);
        EXPECT_EQ(backend.name, "cpu");
    }
}
