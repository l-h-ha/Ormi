#include <cstddef>
#include <cstdint>
#include <limits>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/allocator.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/backend/id.h>

#include "core/runtime/dispatch/device.h"

namespace {
    class CUDAAllocatorTest : public ::testing::Test {
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

    bool memory_pool_supported(int device) {
        int supported = 0;
        return cudaDeviceGetAttribute(
            &supported,
            cudaDevAttrMemoryPoolsSupported,
            device
        ) == cudaSuccess && supported != 0;
    }

    TEST_F(CUDAAllocatorTest, allocatesAndReleasesDeviceMemory) {
        auto* allocator = ormi::core::runtime::device::allocator(ormi::core::Device::from("cuda"));

        ASSERT_NE(allocator, nullptr);
        EXPECT_FALSE(allocator->host_visible());
        EXPECT_EQ(allocator->allocate(0), nullptr);

        constexpr std::size_t bytes = 256;
        void* ptr = allocator->allocate(bytes);

        ASSERT_NE(ptr, nullptr);
        EXPECT_NO_THROW(allocator->deallocate(ptr, bytes));
        EXPECT_NO_THROW(allocator->deallocate(nullptr, 0));
    }

    TEST_F(CUDAAllocatorTest, cachesAllocatorPerDevice) {
        const auto device = ormi::core::Device::from("cuda");
        auto* first = ormi::core::runtime::device::allocator(device);
        auto* second = ormi::core::runtime::device::allocator(device);

        ASSERT_NE(first, nullptr);
        EXPECT_EQ(second, first);
    }

    TEST_F(CUDAAllocatorTest, bufferOwnsDeviceMemory) {
        constexpr std::size_t bytes = 256;
        ormi::core::memory::Buffer buffer(ormi::core::Device::from("cuda"), bytes);

        EXPECT_EQ(buffer.bytes(), bytes);
        EXPECT_NE(buffer.data(), nullptr);
        EXPECT_NE(buffer.allocator(), nullptr);
        EXPECT_EQ(buffer.device().name(), "cuda");
    }

    TEST_F(CUDAAllocatorTest, rejectsUnavailableDeviceIndex) {
        const auto unavailable = ormi::core::Device::from("cuda", device_count_);

        EXPECT_THROW(
            (void)ormi::core::runtime::device::allocator(unavailable),
            ormi::BackendError
        );
    }

    TEST_F(CUDAAllocatorTest, accountsLiveAllocationInPrivatePool) {
        const auto device = ormi::core::Device::from("cuda");

        if (!memory_pool_supported(device.index())) {
            GTEST_SKIP() << "CUDA memory pools are unavailable.";
        }

        auto* allocator = ormi::core::runtime::device::allocator(device);
        allocator->empty_cache();

        constexpr std::size_t bytes = 1024 * 1024;
        cudaMemPool_t pool = nullptr;

        {
            ormi::core::memory::Buffer buffer(device, bytes);
            ormi::core::runtime::device::synchronize(device);

            CUmemoryPool allocation_pool = nullptr;
            ASSERT_EQ(
                cuPointerGetAttribute(
                    &allocation_pool,
                    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                    reinterpret_cast<CUdeviceptr>(buffer.data())
                ),
                CUDA_SUCCESS
            );
            ASSERT_NE(allocation_pool, nullptr);
            pool = reinterpret_cast<cudaMemPool_t>(allocation_pool);

            std::uint64_t used_during = 0;
            ASSERT_EQ(
                cudaMemPoolGetAttribute(
                    pool,
                    cudaMemPoolAttrUsedMemCurrent,
                    &used_during
                ),
                cudaSuccess
            );
            EXPECT_GE(used_during, bytes);
        }

        ormi::core::runtime::device::synchronize(device);

        std::uint64_t used_after = 0;
        ASSERT_EQ(
            cudaMemPoolGetAttribute(
                pool,
                cudaMemPoolAttrUsedMemCurrent,
                &used_after
            ),
            cudaSuccess
        );
        EXPECT_EQ(used_after, 0u);
    }

    TEST_F(CUDAAllocatorTest, emptyCacheTrimsReclaimablePoolMemory) {
        const auto device = ormi::core::Device::from("cuda");

        if (!memory_pool_supported(device.index())) {
            GTEST_SKIP() << "CUDA memory pools are unavailable.";
        }

        auto* allocator = ormi::core::runtime::device::allocator(device);
        constexpr std::size_t bytes = 8 * 1024 * 1024;
        cudaMemPool_t pool = nullptr;

        {
            ormi::core::memory::Buffer buffer(device, bytes);
            ormi::core::runtime::device::synchronize(device);

            CUmemoryPool allocation_pool = nullptr;
            ASSERT_EQ(
                cuPointerGetAttribute(
                    &allocation_pool,
                    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                    reinterpret_cast<CUdeviceptr>(buffer.data())
                ),
                CUDA_SUCCESS
            );
            ASSERT_NE(allocation_pool, nullptr);
            pool = reinterpret_cast<cudaMemPool_t>(allocation_pool);
        }

        ormi::core::runtime::device::synchronize(device);

        std::uint64_t reserved_before = 0;
        ASSERT_EQ(
            cudaMemPoolGetAttribute(
                pool,
                cudaMemPoolAttrReservedMemCurrent,
                &reserved_before
            ),
            cudaSuccess
        );
        EXPECT_GT(reserved_before, 0u);

        allocator->empty_cache();

        std::uint64_t reserved_after = 0;
        ASSERT_EQ(
            cudaMemPoolGetAttribute(
                pool,
                cudaMemPoolAttrReservedMemCurrent,
                &reserved_after
            ),
            cudaSuccess
        );
        EXPECT_LT(reserved_after, reserved_before);
    }
}
