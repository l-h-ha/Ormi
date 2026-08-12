#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>

#include "core/runtime/dispatch/device.h"

namespace {
    class CUDAMemoryTest : public ::testing::Test {
    protected:
        void SetUp() override {
            const cudaError_t status = cudaGetDeviceCount(&device_count_);

            if (status != cudaSuccess) {
                GTEST_SKIP() << "CUDA runtime is unavailable: " << cudaGetErrorString(status);
            }

            if (device_count_ == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
        }

        int device_count_ = 0;
    };

    TEST_F(CUDAMemoryTest, copiesHostToDeviceAndBack) {
        const std::vector<std::uint32_t> expected{
            0x00000000u,
            0x12345678u,
            0xabcdef01u,
            0xffffffffu
        };
        const std::size_t bytes = expected.size() * sizeof(expected.front());

        ormi::core::memory::Buffer host_input(ormi::core::Device::cpu(), bytes);
        ormi::core::memory::Buffer device(ormi::core::Device::from("cuda"), bytes);
        ormi::core::memory::Buffer host_output(ormi::core::Device::cpu(), bytes);

        std::memcpy(host_input.data(), expected.data(), bytes);
        ormi::core::memory::copy(device, host_input);
        ormi::core::memory::copy(host_output, device);

        EXPECT_EQ(std::memcmp(host_output.data(), expected.data(), bytes), 0);
    }

    TEST_F(CUDAMemoryTest, copiesBetweenBuffersOnSameDevice) {
        const std::vector<std::uint32_t> expected{
            0x10203040u,
            0x50607080u,
            0x90a0b0c0u,
            0xd0e0f000u
        };
        const std::size_t bytes = expected.size() * sizeof(expected.front());

        ormi::core::memory::Buffer host_input(ormi::core::Device::cpu(), bytes);
        ormi::core::memory::Buffer first(ormi::core::Device::from("cuda"), bytes);
        ormi::core::memory::Buffer second(ormi::core::Device::from("cuda"), bytes);
        ormi::core::memory::Buffer host_output(ormi::core::Device::cpu(), bytes);

        std::memcpy(host_input.data(), expected.data(), bytes);
        ormi::core::memory::copy(first, host_input);
        ormi::core::memory::copy(second, first);
        ormi::core::memory::copy(host_output, second);

        EXPECT_EQ(std::memcmp(host_output.data(), expected.data(), bytes), 0);
    }

    TEST_F(CUDAMemoryTest, fillsDeviceBytes) {
        constexpr std::size_t bytes = 257;
        constexpr unsigned char value = 0xa5;

        ormi::core::memory::Buffer device(ormi::core::Device::from("cuda"), bytes);
        ormi::core::memory::Buffer host_output(ormi::core::Device::cpu(), bytes);

        ormi::core::memory::fill(device, value);
        ormi::core::memory::copy(host_output, device);

        const auto* output = static_cast<const unsigned char*>(host_output.data());
        for (std::size_t i = 0; i < bytes; ++i) {
            EXPECT_EQ(output[i], value) << "byte " << i;
        }
    }

    TEST_F(CUDAMemoryTest, ordersByteAndTypedFillsOnDeviceStream) {
        constexpr std::size_t elements = 1024;
        constexpr std::size_t bytes = elements * sizeof(float);
        constexpr float expected = 3.25f;
        const auto device_id = ormi::core::Device::from("cuda");
        ormi::core::memory::Buffer device(device_id, bytes);
        ormi::core::memory::Buffer host(ormi::core::Device::cpu(), bytes);

        ormi::core::memory::fill(device, 0);
        ormi::core::memory::fill_scalar(
            device.data(),
            device_id,
            &expected,
            elements,
            ormi::core::DType::float32
        );
        ormi::core::runtime::device::synchronize(device_id);
        ormi::core::memory::copy(host, device);

        const auto* output = static_cast<const float*>(host.data());
        for (std::size_t i = 0; i < elements; ++i) {
            EXPECT_FLOAT_EQ(output[i], expected);
        }
    }

    TEST_F(CUDAMemoryTest, zeroByteOperationsAcceptNullPointers) {
        EXPECT_NO_THROW(ormi::core::memory::copy(
            nullptr,
            ormi::core::Device::from("cuda"),
            nullptr,
            ormi::core::Device::cpu(),
            0
        ));
        EXPECT_NO_THROW(ormi::core::memory::fill(
            nullptr,
            ormi::core::Device::from("cuda"),
            0xff,
            0
        ));
    }

    TEST_F(CUDAMemoryTest, synchronizesSelectedDevice) {
        EXPECT_NO_THROW(
            ormi::core::runtime::device::synchronize(ormi::core::Device::from("cuda"))
        );
    }

    TEST_F(CUDAMemoryTest, rejectsUnavailableDeviceDuringSynchronization) {
        EXPECT_THROW(
            ormi::core::runtime::device::synchronize(
                ormi::core::Device::from("cuda", device_count_)
            ),
            ormi::BackendError
        );
    }

    TEST_F(CUDAMemoryTest, copiesBetweenCUDADevices) {
        if (device_count_ < 2) {
            GTEST_SKIP() << "Cross-device copy requires at least two CUDA devices.";
        }

        constexpr std::uint32_t expected = 0x12345678u;
        constexpr std::size_t bytes = sizeof(expected);

        ormi::core::memory::Buffer host_input(ormi::core::Device::cpu(), bytes);
        ormi::core::memory::Buffer first(ormi::core::Device::from("cuda", 0), bytes);
        ormi::core::memory::Buffer second(ormi::core::Device::from("cuda", 1), bytes);
        ormi::core::memory::Buffer host_output(ormi::core::Device::cpu(), bytes);

        std::memcpy(host_input.data(), &expected, bytes);
        ormi::core::memory::copy(first, host_input);
        ormi::core::memory::copy(second, first);
        ormi::core::memory::copy(host_output, second);

        EXPECT_EQ(std::memcmp(host_output.data(), &expected, bytes), 0);
    }
}
