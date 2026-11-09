#include <algorithm>
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
            std::vector<int> architectures(static_cast<std::size_t>(count));
            ASSERT_EQ(nvrtcGetSupportedArchs(architectures.data()), NVRTC_SUCCESS);
            const auto architecture = *std::min_element(architectures.begin(), architectures.end());
            major_ = architecture / 10;
            minor_ = architecture % 10;
        }

        int major_ = 0;
        int minor_ = 0;
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
}
