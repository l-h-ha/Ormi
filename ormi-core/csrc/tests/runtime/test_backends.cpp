#include <cstddef>
#include <span>

#include <gtest/gtest.h>

#include <ormi/backend/api.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/backend/id.h>

#include "core/runtime/backend.h"
#include "core/runtime/backends.h"

namespace {
    ormi::core::memory::Allocator* test_allocator(ormi::core::Device) {
        return nullptr;
    }

    void test_sync(ormi::core::Device) {}

    void test_copy(
        ormi::core::runtime::Transfer,
        ormi::core::Device,
        void*,
        ormi::core::Device,
        const void*,
        std::size_t
    ) {}

    void test_copy_async(
        ormi::core::Device,
        void*,
        const void*,
        std::size_t
    ) {}

    void test_fill(ormi::core::Device, void*, unsigned char, std::size_t) {}

    void test_fill_scalar(
        ormi::core::Device,
        void*,
        const void*,
        std::size_t,
        ormi::core::DType
    ) {}

    void test_prepare_many(
        std::span<ormi::core::runtime::FusionJob>
    ) {}

    ormi::core::runtime::Backend minimal_backend() {
        return {
            .abi = ormi::core::runtime::backend_abi,
            .memory = {
                .allocator = test_allocator,
                .synchronize = test_sync,
                .copy = test_copy,
                .direct_copy = nullptr,
                .copy_async = test_copy_async,
                .fill = test_fill,
                .fill_scalar = test_fill_scalar
            }
        };
    }

    TEST(BackendsTest, staticInstanceInitializesCPUBackend) {
        auto& backends = ormi::core::runtime::Backends::instance();
        const auto cpu_id = backends.find("cpu");

        ASSERT_NE(cpu_id, ormi::core::runtime::backend_invalid);
        EXPECT_TRUE(backends.available(cpu_id));

        const auto& cpu = backends.require(cpu_id);
        EXPECT_EQ(cpu.abi, ormi::core::runtime::backend_abi);
        EXPECT_NE(cpu.memory.allocator, nullptr);
        EXPECT_NE(cpu.memory.synchronize, nullptr);
        EXPECT_NE(cpu.memory.copy, nullptr);
        EXPECT_EQ(cpu.memory.direct_copy, nullptr);
        EXPECT_NE(cpu.memory.copy_async, nullptr);
        EXPECT_NE(cpu.memory.fill, nullptr);
        EXPECT_NE(cpu.memory.fill_scalar, nullptr);
        EXPECT_NE(cpu.ops.cast, nullptr);
        EXPECT_NE(cpu.ops.sgd, nullptr);
        EXPECT_NE(cpu.ops.matmul, nullptr);
        EXPECT_NE(cpu.ops.reduce, nullptr);
        EXPECT_NE(cpu.ops.all_finite, nullptr);
        EXPECT_NE(cpu.ops.binary, nullptr);
        EXPECT_NE(cpu.ops.unary, nullptr);
        EXPECT_NE(cpu.fusion.prepare_many, nullptr);
    }

    TEST(BackendsTest, resolvesBackendNames) {
        auto& backends = ormi::core::runtime::Backends::instance();
        const auto cpu = backends.find("cpu");
        const auto cuda = backends.find("cuda");

        EXPECT_NE(cpu, ormi::core::runtime::backend_invalid);
        EXPECT_NE(cuda, ormi::core::runtime::backend_invalid);
        EXPECT_NE(cpu, cuda);
        EXPECT_EQ(ormi::core::runtime::catalog::host(), cpu);
        EXPECT_EQ(backends.find(""), ormi::core::runtime::backend_invalid);
        EXPECT_EQ(backends.find("CPU"), ormi::core::runtime::backend_invalid);
        EXPECT_EQ(backends.find("metal"), ormi::core::runtime::backend_invalid);
    }

    TEST(BackendsTest, rejectsInvalidBackendId) {
        auto& backends = ormi::core::runtime::Backends::instance();

        EXPECT_FALSE(backends.available(ormi::core::runtime::backend_invalid));
        EXPECT_THROW(
            (void)backends.require(ormi::core::runtime::backend_invalid),
            ormi::DeviceError
        );
    }

    TEST(BackendsTest, fillsUnsupportedOperationSlotsOnce) {
        const auto backend = ormi::core::runtime::normalize(
            minimal_backend(),
            "minimal"
        );

        EXPECT_NE(backend.ops.binary, nullptr);
        EXPECT_NE(backend.ops.unary, nullptr);
        EXPECT_NE(backend.ops.cast, nullptr);
        EXPECT_NE(backend.ops.matmul, nullptr);
        EXPECT_NE(backend.ops.reduce, nullptr);
        EXPECT_NE(backend.ops.all_finite, nullptr);
        EXPECT_NE(backend.ops.sgd, nullptr);
        EXPECT_NE(backend.fractional.execute, nullptr);
        EXPECT_EQ(backend.fractional.prepare, nullptr);
        EXPECT_EQ(backend.fusion.execute, nullptr);
        EXPECT_EQ(backend.recording.record, nullptr);
        EXPECT_THROW(
            (void)backend.ops.all_finite({}),
            ormi::OperationError
        );
    }

    TEST(BackendsTest, requiresOnlyCompleteMemoryFunctions) {
        auto backend = minimal_backend();
        backend.memory.copy = nullptr;

        EXPECT_THROW(
            (void)ormi::core::runtime::normalize(backend, "minimal"),
            ormi::DeviceError
        );
    }

    TEST(BackendsTest, rejectsBatchPreparationWithoutFusionExecution) {
        auto backend = minimal_backend();
        backend.fusion.prepare_many = test_prepare_many;

        EXPECT_THROW(
            (void)ormi::core::runtime::normalize(backend, "minimal"),
            ormi::DeviceError
        );
    }
}
