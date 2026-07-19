#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/backend_id.h>

namespace {
    TEST(DeviceTest, cpuDeviceResolvesBuiltinBackend) {
        const auto device = ormi::core::Device::cpu();

        EXPECT_EQ(device.backend_id(), ormi::core::runtime::backend_id_cpu);
        EXPECT_EQ(device.index(), 0);
        ASSERT_NE(device.backend(), nullptr);
        EXPECT_EQ(device.backend()->name, "cpu");
        EXPECT_TRUE(device.is_cpu());
        EXPECT_FALSE(device.is_cuda());
    }

#ifdef ORMI_ENABLE_CUDA
    TEST(DeviceTest, cudaDeviceResolvesOptionalBackend) {
        const auto device = ormi::core::Device::cuda(1);

        EXPECT_EQ(device.backend_id(), ormi::core::runtime::backend_id_cuda);
        EXPECT_EQ(device.index(), 1);
        ASSERT_NE(device.backend(), nullptr);
        EXPECT_EQ(device.backend()->name, "cuda");
        EXPECT_FALSE(device.is_cpu());
        EXPECT_TRUE(device.is_cuda());
    }
#else
    TEST(DeviceTest, rejectsCudaDeviceWhenDisabled) {
        EXPECT_THROW(ormi::core::Device::cuda(), ormi::DeviceError);
    }
#endif

    TEST(DeviceTest, parsesCpuDeviceString) {
        const auto device = ormi::core::Device::from("cpu");

        EXPECT_EQ(device.backend_id(), ormi::core::runtime::backend_id_cpu);
        EXPECT_EQ(device.index(), 0);
    }

#ifdef ORMI_ENABLE_CUDA
    TEST(DeviceTest, parsesCudaDeviceStringWhenEnabled) {
        const auto default_device = ormi::core::Device::from("cuda");
        EXPECT_EQ(default_device.backend_id(), ormi::core::runtime::backend_id_cuda);
        EXPECT_EQ(default_device.index(), 0);

        const auto device = ormi::core::Device::from("cuda:2");

        EXPECT_EQ(device.backend_id(), ormi::core::runtime::backend_id_cuda);
        EXPECT_EQ(device.index(), 2);
        ASSERT_NE(device.backend(), nullptr);
        EXPECT_EQ(device.backend()->name, "cuda");
    }
#else
    TEST(DeviceTest, rejectsCudaDeviceStringWhenDisabled) {
        try {
            (void)ormi::core::Device::from("cuda");
            FAIL() << "Expected DeviceError.";
        } catch (const ormi::DeviceError& error) {
            EXPECT_NE(std::string(error.what()).find("Cannot load backend 'cuda'"), std::string::npos);
        }

        try {
            (void)ormi::core::Device::from("cuda:0");
            FAIL() << "Expected DeviceError.";
        } catch (const ormi::DeviceError& error) {
            EXPECT_NE(std::string(error.what()).find("Cannot load backend 'cuda'"), std::string::npos);
        }
    }
#endif

    TEST(DeviceTest, rejectsInvalidCpuIndex) {
        EXPECT_THROW(ormi::core::Device::from("cpu:1"), ormi::DeviceError);
        EXPECT_THROW(ormi::core::Device(ormi::core::runtime::backend_id_cpu, 1), ormi::DeviceError);
    }

    TEST(DeviceTest, rejectsMalformedDeviceStrings) {
        EXPECT_THROW(ormi::core::Device::from(""), ormi::DeviceError);
        EXPECT_THROW(ormi::core::Device::from(":0"), ormi::DeviceError);
        EXPECT_THROW(ormi::core::Device::from("cuda:"), ormi::DeviceError);
        EXPECT_THROW(ormi::core::Device::from("cuda:abc"), ormi::DeviceError);
        EXPECT_THROW(ormi::core::Device::from("cuda:1x"), ormi::DeviceError);
        EXPECT_THROW(ormi::core::Device::from("metal:0"), ormi::DeviceError);
    }
}
