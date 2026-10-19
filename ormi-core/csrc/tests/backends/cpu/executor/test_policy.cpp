#include <array>
#include <chrono>
#include <string>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/executor.h"
#include "platform/platform.h"

namespace {
    namespace executor = ormi::backends::cpu::executor;

    void check_process(const char* mode, int expected_code) {
        const std::array<std::string, 1> arguments{mode};
        const auto result = ormi::platform::run(
            ORMI_EXECUTOR_HELPER, arguments, {}, std::chrono::seconds(10)
        );
        EXPECT_EQ(result.state, ormi::platform::Process::State::exited) << result.output;
        EXPECT_EQ(result.code, expected_code) << result.output;
    }

    TEST(CPUExecutorPolicyTest, keepsSmallWorkSerial) {
        EXPECT_EQ(executor::slots(1024, 1, 512 * 1024 - 1), 1u);
        EXPECT_EQ(executor::slots(1024, 1, 512 * 1024), 3u);
    }

    TEST(CPUExecutorPolicyTest, boundsParticipantsByUnitsAndConfiguredThreads) {
        EXPECT_EQ(executor::slots(0, 1, 1024 * 1024), 1u);
        EXPECT_EQ(executor::slots(1, 1, 1024 * 1024), 1u);
        EXPECT_EQ(executor::slots(8, 4, 1024 * 1024), 2u);
        EXPECT_EQ(executor::slots(9, 4, 1024 * 1024), 3u);
        EXPECT_EQ(executor::slots(1024, 1, 1024 * 1024), 4u);
        EXPECT_THROW(executor::slots(1024, 0, 0), ormi::OperationError);
    }

    TEST(CPUExecutorPolicyTest, validatesConfigurationLazilyAndCachesOnlySuccess) {
        check_process("lazy", 0);
    }

    TEST(CPUExecutorPolicyTest, respectsSingleThreadConfiguration) {
        check_process("single", 0);
    }

    TEST(CPUExecutorPolicyTest, initializesOnceWithConcurrentFirstSubmissions) {
        check_process("startup", 0);
    }

    TEST(CPUExecutorPolicyTest, escapingWorkerExceptionTerminatesProcess) {
        check_process("worker-throws", 73);
    }
}
