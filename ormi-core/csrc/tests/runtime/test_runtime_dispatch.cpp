#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/runtime/dispatch/device.h>

namespace {
    TEST(BackendDispatchTest, dispatchesThroughResolvedCPUBackend) {
        const auto device = ormi::core::Device::cpu();

        EXPECT_NE(ormi::core::runtime::device::allocator(device), nullptr);
        EXPECT_NO_THROW(ormi::core::runtime::device::synchronize(device));
    }

#ifdef ORMI_ENABLE_CUDA
    TEST(BackendDispatchTest, dispatchesThroughResolvedCUDABackendWhenEnabled) {
        const auto device = ormi::core::Device::cuda();

        EXPECT_NE(ormi::core::runtime::device::allocator(device), nullptr);
        EXPECT_NO_THROW(ormi::core::runtime::device::synchronize(device));
    }
#endif
}
