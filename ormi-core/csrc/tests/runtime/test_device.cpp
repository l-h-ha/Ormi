#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
namespace {
    TEST(DeviceTest, cpuDeviceResolvesBuiltinBackend) {
        const auto device = ormi::core::Device::cpu();

        EXPECT_EQ(device.index(), 0);
        EXPECT_EQ(device.name(), "cpu");
        EXPECT_FALSE(device.indexed());
        EXPECT_TRUE(device.is_host());
    }

#ifdef ORMI_ENABLE_CUDA
    TEST(DeviceTest, cudaDeviceResolvesOptionalBackend) {
        const auto device = ormi::core::Device::from("cuda", 1);

        EXPECT_EQ(device.index(), 1);
        EXPECT_EQ(device.name(), "cuda");
        EXPECT_TRUE(device.indexed());
        EXPECT_FALSE(device.is_host());
    }
#else
    TEST(DeviceTest, rejectsCudaDeviceWhenDisabled) {
        EXPECT_THROW(ormi::core::Device::from("cuda"), ormi::DeviceError);
    }
#endif

    TEST(DeviceTest, parsesCpuDeviceString) {
        const auto device = ormi::core::Device::from("cpu");

        EXPECT_EQ(device.backend_id(), ormi::core::Device::cpu().backend_id());
        EXPECT_EQ(device.index(), 0);
    }

#ifdef ORMI_ENABLE_CUDA
    TEST(DeviceTest, parsesCudaDeviceStringWhenEnabled) {
        const auto default_device = ormi::core::Device::from("cuda");
        EXPECT_EQ(
            default_device.backend_id(),
            ormi::core::Device::from("cuda").backend_id()
        );
        EXPECT_EQ(default_device.index(), 0);

        const auto device = ormi::core::Device::from("cuda:2");

        EXPECT_EQ(device.backend_id(), default_device.backend_id());
        EXPECT_EQ(device.index(), 2);
        EXPECT_EQ(device.name(), "cuda");
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
        EXPECT_THROW(
            ormi::core::Device(ormi::core::Device::cpu().backend_id(), 1),
            ormi::DeviceError
        );
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
