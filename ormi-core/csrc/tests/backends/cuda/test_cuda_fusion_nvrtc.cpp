#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nvrtc.h>

#include <ormi/core/error.h>

#include "backends/cuda/fusion/nvrtc.h"

namespace {
    namespace rtc = ormi::backends::cuda::fusion::nvrtc;

    constexpr char source[] = R"(
#if __cplusplus != 201703L
#error ORMI_EXPECTED_CXX17
#endif
extern "C" __global__ void ormi_nvrtc_test(float* values) {
    if constexpr (sizeof(float) == 4) {
        values[threadIdx.x] = 2.0f;
    }
}
)";

    class CUDAFusionNVRTCTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int count = 0;

            ASSERT_EQ(nvrtcGetNumSupportedArchs(&count), NVRTC_SUCCESS);
            ASSERT_GT(count, 0);
            architectures_.resize(static_cast<std::size_t>(count));

            ASSERT_EQ(nvrtcGetSupportedArchs(architectures_.data()), NVRTC_SUCCESS);
            std::sort(architectures_.begin(), architectures_.end());
            const auto architecture = architectures_.front();
            major_ = architecture / 10;
            minor_ = architecture % 10;
        }

        int major_ = 0;
        int minor_ = 0;
        std::vector<int> architectures_;
    };

    TEST_F(CUDAFusionNVRTCTest, contextPreservesArchitectureAndCompilerVersion) {
        int major = 0;
        int minor = 0;

        ASSERT_EQ(nvrtcVersion(&major, &minor), NVRTC_SUCCESS);
        EXPECT_EQ(rtc::context(major_, minor_),
            "compute_" + std::to_string(major_) + std::to_string(minor_) +
            "\nnvrtc_" + std::to_string(major) + "." + std::to_string(minor));
    }

    TEST_F(CUDAFusionNVRTCTest, compilesCXX17KernelToPTX) {
        const auto ptx = rtc::compile(source, major_, minor_);

        ASSERT_FALSE(ptx.empty());
        EXPECT_EQ(ptx.back(), '\0');
        EXPECT_NE(ptx.find(".entry ormi_nvrtc_test"), std::string::npos);
    }

    TEST_F(CUDAFusionNVRTCTest, identicalSourceProducesIdenticalPTX) {
        const auto first = rtc::compile(source, major_, minor_);

        EXPECT_EQ(rtc::compile(source, major_, minor_), first);
    }

    TEST_F(CUDAFusionNVRTCTest, invalidSourceReportsDiagnosticsAndCanRetry) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            try {
                (void)rtc::compile("#error ORMI_NVRTC_TEST_FAILURE\n", major_, minor_);
                FAIL() << "Invalid CUDA source was accepted.";
            } catch (const ormi::jitError& error) {
                const std::string message(error.what());

                EXPECT_NE(message.find("CUDA fusion compilation failed:"), std::string::npos);
                EXPECT_NE(message.find("ORMI_NVRTC_TEST_FAILURE"), std::string::npos);
                EXPECT_NE(message.find("ormi_fusion.cu"), std::string::npos);
            }

            EXPECT_FALSE(rtc::compile(source, major_, minor_).empty());
        }
    }

    TEST_F(CUDAFusionNVRTCTest, invalidArchitectureCanRetry) {
        EXPECT_THROW((void)rtc::compile(source, -1, 0), ormi::jitError);
        EXPECT_FALSE(rtc::compile(source, major_, minor_).empty());
    }

    TEST_F(CUDAFusionNVRTCTest, supportedTargetsRemainUnchanged) {
        for (const auto architecture : architectures_) {
            EXPECT_EQ(rtc::target(architecture / 10, architecture % 10), architecture);
        }
    }

    TEST_F(CUDAFusionNVRTCTest, selectsHighestSupportedTargetForGapsAndNewerGPUs) {
        for (int requested = architectures_.front(); requested <= architectures_.back() + 10; ++requested) {
            const auto next = std::upper_bound(architectures_.begin(), architectures_.end(), requested);

            ASSERT_NE(next, architectures_.begin());
            EXPECT_EQ(rtc::target(requested / 10, requested % 10), *(next - 1));
        }
    }

    TEST_F(CUDAFusionNVRTCTest, rejectsGPUsOlderThanTheCompilerSupports) {
        const auto requested = architectures_.front() - 1;

        EXPECT_THROW((void)rtc::target(requested / 10, requested % 10), ormi::jitError);
    }

    TEST_F(CUDAFusionNVRTCTest, rejectsInvalidComputeCapabilities) {
        EXPECT_THROW((void)rtc::target(-1, 0), ormi::jitError);
        EXPECT_THROW((void)rtc::target(0, 0), ormi::jitError);
        EXPECT_THROW((void)rtc::target(7, -1), ormi::jitError);
        EXPECT_THROW((void)rtc::target(7, 10), ormi::jitError);
        EXPECT_EQ(rtc::target(std::numeric_limits<int>::max(), 9), architectures_.back());
    }

    TEST_F(CUDAFusionNVRTCTest, newerGPUsUseTheSelectedTargetInTheDiskCacheContext) {
        const auto selected = architectures_.back();
        const auto future_major = selected / 10 + 1;

        EXPECT_EQ(rtc::context(future_major, 0), rtc::context(selected / 10, selected % 10));
    }

    TEST_F(CUDAFusionNVRTCTest, newerGPUsCompileIdenticalPTXToTheirSelectedTarget) {
        const auto selected = architectures_.back();
        const auto future_major = selected / 10 + 1;
        const auto ptx = rtc::compile(source, future_major, 0);

        EXPECT_NE(ptx.find(".target sm_" + std::to_string(selected)), std::string::npos);
        EXPECT_EQ(ptx, rtc::compile(source, selected / 10, selected % 10));
    }
}
