#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "platform/platform.h"

namespace {
    const std::filesystem::path helper(ORMI_PROCESS_HELPER);
#if defined(_WIN32)
    constexpr const char* newline = "\r\n";
#else
    constexpr const char* newline = "\n";
#endif

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

    TEST(PlatformProcessTest, preservesArgumentBoundariesWithoutAShell) {
        const std::vector<std::string> arguments{
            "arguments", "two words", "a\tb", "quote\"inside",
            "trailing slash\\", ";$()&*"
        };
        std::string expected;
        for (std::size_t i = 1; i < arguments.size(); ++i) {
            expected += std::to_string(arguments[i].size()) + ":" +
                arguments[i] + newline;
        }
        const auto result = ormi::platform::run(helper, arguments);
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 0);
        EXPECT_EQ(result.output, expected);
    }

    TEST(PlatformProcessTest, suppliesExplicitEnvironment) {
        const auto result = ormi::platform::run(
            helper,
            std::vector<std::string>{"environment", "ORMI_PROCESS_TEST_VALUE"},
            std::vector<std::string>{"ORMI_PROCESS_TEST_VALUE=two words=one value"}
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 0);
        EXPECT_EQ(result.output, "two words=one value");
    }

    TEST(PlatformProcessTest, inheritsEnvironmentWhenNotReplaced) {
        const auto* value = std::getenv("PATH");
        const std::string expected(value ? value : "<unset>");
        const auto result = ormi::platform::run(
            helper, std::vector<std::string>{"environment", "PATH"}
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 0);
        EXPECT_EQ(result.output, expected);
    }

    TEST(PlatformProcessTest, capturesBothOutputStreams) {
        const auto result = ormi::platform::run(
            helper, std::vector<std::string>{"streams"}
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 0);
        EXPECT_EQ(result.output, std::string("stdout") + newline + "stderr" + newline);
    }

    TEST(PlatformProcessTest, drainsOutputLargerThanPipeCapacity) {
        const auto result = ormi::platform::run(
            helper, std::vector<std::string>{"large"}, {},
            std::chrono::seconds(10)
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 17);
        EXPECT_EQ(result.output,
            std::string(128 * 1024, 'x') + std::string(128 * 1024, 'y'));
    }

    TEST(PlatformProcessTest, handlesExitWithoutOutput) {
        const auto result = ormi::platform::run(
            helper, std::vector<std::string>{"quiet"}
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 0);
        EXPECT_TRUE(result.output.empty());
    }

    TEST(PlatformProcessTest, keepsConcurrentCallsIndependent) {
        std::vector<std::future<ormi::platform::Process>> calls;
        for (unsigned i = 0; i < 8; ++i) {
            calls.push_back(std::async(std::launch::async, [i] {
                return ormi::platform::run(
                    helper,
                    std::vector<std::string>{"environment", "ORMI_PROCESS_TEST_VALUE"},
                    std::vector<std::string>{"ORMI_PROCESS_TEST_VALUE=" + std::to_string(i)}
                );
            }));
        }
        for (std::size_t i = 0; i < calls.size(); ++i) {
            const auto result = calls[i].get();
            EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
            EXPECT_EQ(result.code, 0);
            EXPECT_EQ(result.output, std::to_string(i));
        }
    }
}
