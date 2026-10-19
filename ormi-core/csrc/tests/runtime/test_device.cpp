#include <string>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>

namespace {
    TEST(DeviceTest, cpuIdentityMatchesEveryConstructionPath) {
        const auto device = ormi::core::Device::cpu();
        const ormi::core::Device default_device;
        const auto named = ormi::core::Device::from("cpu");
        const auto indexed = ormi::core::Device::from("cpu:0");
        const ormi::core::Device explicit_device(device.backend_id(), 0);

        static_assert(noexcept(device == default_device));
        static_assert(noexcept(device != default_device));
        EXPECT_EQ(device, device);
        EXPECT_EQ(device, default_device);
        EXPECT_EQ(default_device, device);
        EXPECT_EQ(device, named);
        EXPECT_EQ(device, indexed);
        EXPECT_EQ(device, explicit_device);
        EXPECT_FALSE(device != indexed);
    }

    TEST(DeviceTest, copyingAndAssignmentPreserveIdentity) {
        const auto device = ormi::core::Device::cpu();
        const auto copied = device;
        auto assigned = ormi::core::Device::from("cpu:0");
        assigned = copied;

        EXPECT_EQ(copied, device);
        EXPECT_EQ(assigned, copied);
    }

    TEST(DeviceTest, cpuDeviceResolvesBuiltinBackend) {
        const auto device = ormi::core::Device::cpu();

        EXPECT_EQ(device.index(), 0);
        EXPECT_EQ(device.name(), "cpu");
        EXPECT_FALSE(device.indexed());
        EXPECT_TRUE(device.is_host());
    }

#ifdef ORMI_ENABLE_CUDA
    TEST(DeviceTest, differentBackendsAreNotEqualAtTheSameIndex) {
        const auto cpu = ormi::core::Device::cpu();
        const auto cuda = ormi::core::Device::from("cuda:0");

        ASSERT_EQ(cpu.index(), cuda.index());
        EXPECT_NE(cpu, cuda);
        EXPECT_NE(cuda, cpu);
        EXPECT_FALSE(cpu == cuda);
    }

    TEST(DeviceTest, cudaIdentityIncludesTheDeviceIndex) {
        const auto first = ormi::core::Device::from("cuda");
        const auto same = ormi::core::Device::from("cuda:0");
        const auto second = ormi::core::Device::from("cuda", 1);

        EXPECT_EQ(first, same);
        EXPECT_NE(first, second);
        EXPECT_NE(second, first);
        EXPECT_FALSE(first == second);
    }

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
