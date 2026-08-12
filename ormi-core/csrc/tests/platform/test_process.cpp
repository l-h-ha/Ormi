#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "platform/platform.h"

namespace {
    const std::filesystem::path helper(ORMI_PROCESS_HELPER);

    TEST(PlatformProcessTest, capturesNormalExitAndOutput) {
        const auto result = ormi::platform::run(
            helper,
            std::vector<std::string>{"normal"}
        );

        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 7);
        EXPECT_NE(result.output.find("normal output"), std::string::npos);
    }

    TEST(PlatformProcessTest, timesOutProcessGroupAndPreservesOutput) {
        const auto started = std::chrono::steady_clock::now();
        const auto result = ormi::platform::run(
            helper,
            std::vector<std::string>{"sleep"},
            {},
            std::chrono::milliseconds(500)
        );
        const auto elapsed = std::chrono::steady_clock::now() - started;

        EXPECT_EQ(result.state, ormi::platform::Process::State::timed_out);
        EXPECT_NE(result.output.find("before timeout"), std::string::npos);
        EXPECT_LT(elapsed, std::chrono::seconds(2));
    }

    TEST(PlatformProcessTest, reportsAbnormalTermination) {
        const auto result = ormi::platform::run(
            helper,
            std::vector<std::string>{"crash"},
            {},
            std::chrono::seconds(5)
        );

        EXPECT_EQ(result.state, ormi::platform::Process::State::crashed);
        EXPECT_NE(result.output.find("before crash"), std::string::npos);
    }
}
