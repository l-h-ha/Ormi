#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include <sys/wait.h>

#include <gtest/gtest.h>

#include "platform/platform.h"

namespace {
    const std::filesystem::path helper(ORMI_PROCESS_HELPER);

    std::size_t descriptor_count() {
        std::size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
            static_cast<void>(entry);
            ++count;
        }
        return count;
    }

    TEST(PlatformLinuxProcessTest, preservesEmptyArguments) {
        const auto result = ormi::platform::run(
            helper, std::vector<std::string>{"arguments", "", "middle", ""}
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 0);
        EXPECT_EQ(result.output, "0:\n6:middle\n0:\n");
    }

    TEST(PlatformLinuxProcessTest, failedExecRetainsExit127) {
        const auto result = ormi::platform::run(
            helper / "not-an-executable", std::vector<std::string>{}
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited);
        EXPECT_EQ(result.code, 127);
        EXPECT_TRUE(result.output.empty());
    }

    TEST(PlatformLinuxProcessTest, timeoutRetainsKillSignalExitCode) {
        const auto result = ormi::platform::run(
            helper, std::vector<std::string>{"sleep"}, {},
            std::chrono::milliseconds(100)
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::timed_out);
        EXPECT_EQ(result.code, SIGKILL);
    }

    TEST(PlatformLinuxProcessTest, reapsExitedAndTerminatedChildren) {
        for (const auto* mode : {"normal", "crash", "sleep"}) {
            const auto timeout = std::string_view(mode) == "sleep"
                ? std::chrono::milliseconds(100)
                : std::chrono::milliseconds(5000);
            const auto result = ormi::platform::run(
                helper, std::vector<std::string>{mode}, {}, timeout
            );
            EXPECT_EQ(result.code, std::string(mode) == "normal" ? 7 : SIGKILL);
            int status = 0;
            errno = 0;
            const auto remaining = waitpid(-1, &status, WNOHANG);
            const auto error = errno;
            EXPECT_EQ(remaining, -1) << mode;
            EXPECT_EQ(error, ECHILD) << mode;
        }
    }

    TEST(PlatformLinuxProcessTest, closesDescriptorsAfterExitCrashAndFailedExec) {
        const auto before = descriptor_count();
        for (unsigned i = 0; i < 12; ++i) {
            ormi::platform::run(helper, std::vector<std::string>{"normal"});
            ormi::platform::run(helper, std::vector<std::string>{"crash"});
            ormi::platform::run(helper / "not-an-executable", std::vector<std::string>{});
        }
        ormi::platform::run(
            helper, std::vector<std::string>{"sleep"}, {},
            std::chrono::milliseconds(100)
        );
        EXPECT_EQ(descriptor_count(), before);
    }
}
