#include <cstdint>
#include <limits>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>

#include "backends/cuda/state.h"

namespace {
    class CUDAStateTest : public ::testing::Test {
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

    TEST_F(CUDAStateTest, cachesOneStatePerDevice) {
        const auto device = ormi::core::Device::cuda();
        auto& first = ormi::backends::cuda::state(device);
        auto& second = ormi::backends::cuda::state(device);

        EXPECT_EQ(&first, &second);
        EXPECT_EQ(first.index(), device.index());
    }

    TEST_F(CUDAStateTest, ownsNonBlockingStream) {
        auto& state = ormi::backends::cuda::state(ormi::core::Device::cuda());

        ASSERT_NE(state.stream(), nullptr);

        unsigned int flags = 0;
        ASSERT_EQ(cudaStreamGetFlags(state.stream(), &flags), cudaSuccess);
        EXPECT_NE(flags & cudaStreamNonBlocking, 0u);
    }

    TEST_F(CUDAStateTest, reportsRuntimePoolCapability) {
        const auto device = ormi::core::Device::cuda();
        auto& state = ormi::backends::cuda::state(device);

        int supported = 0;
        ASSERT_EQ(
            cudaDeviceGetAttribute(
                &supported,
                cudaDevAttrMemoryPoolsSupported,
                device.index()
            ),
            cudaSuccess
        );

    #if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        EXPECT_EQ(state.has_pool(), supported != 0);

        if (state.has_pool()) {
            EXPECT_NE(state.pool(), nullptr);

            cudaMemPool_t default_pool = nullptr;
            ASSERT_EQ(
                cudaDeviceGetDefaultMemPool(&default_pool, device.index()),
                cudaSuccess
            );
            EXPECT_NE(state.pool(), default_pool);

            std::uint64_t release_threshold = 0;
            ASSERT_EQ(
                cudaMemPoolGetAttribute(
                    state.pool(),
                    cudaMemPoolAttrReleaseThreshold,
                    &release_threshold
                ),
                cudaSuccess
            );
            EXPECT_EQ(
                release_threshold,
                std::numeric_limits<std::uint64_t>::max()
            );
        }
    #else
        EXPECT_FALSE(state.has_pool());
    #endif
    }

    TEST_F(CUDAStateTest, rejectsCPUDevice) {
        EXPECT_THROW(
            (void)ormi::backends::cuda::state(ormi::core::Device::cpu()),
            ormi::cudaError
        );
    }

    TEST_F(CUDAStateTest, rejectsUnavailableCUDADevice) {
        EXPECT_THROW(
            (void)ormi::backends::cuda::state(
                ormi::core::Device::cuda(device_count_)
            ),
            ormi::cudaError
        );
    }

    TEST_F(CUDAStateTest, createsStreamOnSelectedDevice) {
        if (device_count_ < 2) {
            GTEST_SKIP() << "Selected-device stream validation requires two CUDA devices.";
        }

        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        auto& second = ormi::backends::cuda::state(ormi::core::Device::cuda(1));
        int stream_device = -1;

        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(second.stream()), cudaSuccess);
        ASSERT_EQ(cudaGetDevice(&stream_device), cudaSuccess);
        EXPECT_EQ(stream_device, 1);
    }
}
