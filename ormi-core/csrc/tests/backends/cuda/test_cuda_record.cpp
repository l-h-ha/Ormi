#include <memory>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/state.h"
#include "core/runtime/device.h"

namespace {
    class CUDARecordingTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            const auto status = cudaGetDeviceCount(&devices);
            if (status != cudaSuccess) {
                GTEST_SKIP() << cudaGetErrorString(status);
            }
            if (devices == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
        }
    };

    TEST_F(CUDARecordingTest, rejectsMissingExecutionBody) {
        const auto device = ormi::core::Device::from("cuda");
        const auto& recording = ormi::core::runtime::backend(device).recording;

        EXPECT_THROW(recording.record(device, nullptr, nullptr), ormi::BackendError);
    }

    TEST_F(CUDARecordingTest, rejectsEmptyReplay) {
        const auto device = ormi::core::Device::from("cuda");
        const auto& recording = ormi::core::runtime::backend(device).recording;

        EXPECT_THROW(recording.replay(nullptr), ormi::BackendError);
    }

    TEST_F(CUDARecordingTest, discardsFailedCaptureAndAllowsAnotherRecording) {
        const auto device = ormi::core::Device::from("cuda");
        const auto output = ormi::core::ops::full({}, 3.0f, device);
        output->realize();
        const auto stream = ormi::backends::cuda::stream(device);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const auto& recording = ormi::core::runtime::backend(device).recording;

        EXPECT_THROW(
            recording.record(device, [](void* context) {
                auto& out = *static_cast<ormi::core::Tensor*>(context);
                ormi::core::memory::zero(out.data(), out.device(), out.bytes());
                ORMI_THROW(ormi::OperationError, "Test capture failed.");
            }, output.get()),
            ormi::OperationError
        );

        cudaStreamCaptureStatus status{};
        ASSERT_EQ(cudaStreamIsCapturing(stream, &status), cudaSuccess);
        EXPECT_EQ(status, cudaStreamCaptureStatusNone);

        std::unique_ptr<void, ormi::core::runtime::ReleaseFn> next(
            recording.record(device, [](void* context) {
                auto& out = *static_cast<ormi::core::Tensor*>(context);
                ormi::core::memory::zero(out.data(), out.device(), out.bytes());
            }, output.get()),
            recording.release
        );
        ASSERT_NE(next, nullptr);
        recording.replay(next.get());
        float value = 3.0f;
        ormi::core::memory::copy(
            &value, ormi::core::Device::cpu(), output->data(), device, sizeof(value)
        );
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}
