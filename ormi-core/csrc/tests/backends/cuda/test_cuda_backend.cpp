#include <gtest/gtest.h>

#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/backend_id.h>
#include <ormi/core/runtime/plugin.h>

extern "C" const ormi::core::runtime::Backend* ormi_backend_init() noexcept;

namespace {
    TEST(CUDABackendTest, exportsCompatibleFunctionTable) {
        const auto* backend = ormi_backend_init();

        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->abi, ormi::core::runtime::backend_abi);
        EXPECT_EQ(backend->id, ormi::core::runtime::backend_id_cuda);
        EXPECT_EQ(backend->name, "cuda");
        EXPECT_NE(backend->allocator, nullptr);
        EXPECT_NE(backend->synchronize, nullptr);
        EXPECT_NE(backend->copy, nullptr);
        EXPECT_NE(backend->fill, nullptr);
        EXPECT_NE(backend->fill_scalar, nullptr);
        EXPECT_NE(backend->cast, nullptr);
        EXPECT_NE(backend->sgd, nullptr);
        EXPECT_NE(backend->matmul, nullptr);
        EXPECT_NE(backend->reduce, nullptr);
        EXPECT_NE(backend->pointwise_binary, nullptr);
        EXPECT_NE(backend->pointwise_unary, nullptr);
    }
}
