#include <gtest/gtest.h>

#include <ormi/backend/api.h>
#include <ormi/backend/plugin.h>

extern "C" const ormi::core::runtime::Backend* ormi_backend_init() noexcept;

namespace {
    TEST(CUDABackendTest, exportsCompatibleFunctionTable) {
        const auto* backend = ormi_backend_init();

        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->abi, ormi::core::runtime::backend_abi);
        EXPECT_NE(backend->memory.allocator, nullptr);
        EXPECT_NE(backend->memory.synchronize, nullptr);
        EXPECT_NE(backend->memory.copy, nullptr);
        EXPECT_EQ(backend->memory.direct_copy, nullptr);
        EXPECT_NE(backend->memory.copy_async, nullptr);
        EXPECT_NE(backend->memory.fill, nullptr);
        EXPECT_NE(backend->memory.fill_scalar, nullptr);
        EXPECT_NE(backend->ops.cast, nullptr);
        EXPECT_NE(backend->ops.sgd, nullptr);
        EXPECT_NE(backend->ops.matmul, nullptr);
        EXPECT_NE(backend->ops.reduce, nullptr);
        EXPECT_NE(backend->ops.binary, nullptr);
        EXPECT_NE(backend->ops.unary, nullptr);
        EXPECT_NE(backend->fractional.execute, nullptr);
        EXPECT_NE(backend->fractional.prepare, nullptr);
        EXPECT_NE(backend->fusion.supports, nullptr);
        EXPECT_NE(backend->fusion.prepare, nullptr);
        EXPECT_NE(backend->fusion.execute, nullptr);
        EXPECT_NE(backend->recording.record, nullptr);
        EXPECT_NE(backend->recording.replay, nullptr);
        EXPECT_NE(backend->recording.release, nullptr);
    }
}
