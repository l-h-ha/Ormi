#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/compiler/probe.h"
#include "platform/platform.h"

namespace {
    namespace jit = ormi::backends::cpu::jit;
    using Process = ormi::platform::Process;

    enum class Failure { none, launch, unexpected };

    struct ProbeCall {
        std::filesystem::path executable;
        std::vector<std::string> arguments;
        std::vector<std::string> environment;
        std::chrono::milliseconds timeout{};
        std::vector<std::filesystem::path> sources;
        std::string source_text;
        Process result{0, "", Process::State::exited};
        Failure failure = Failure::none;
    };

    ProbeCall call;

    void environment(const char* name, const char* value) {
#if defined(_WIN32)
        _putenv_s(name, value ? value : "");
#else
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
#endif
    }

    class EnvironmentGuard {
    public:
        explicit EnvironmentGuard(const char* name) : name_(name) {
            if (const char* value = std::getenv(name)) {
                value_ = value;
            }
        }

        ~EnvironmentGuard() {
            environment(name_, value_ ? value_->c_str() : nullptr);
        }

        EnvironmentGuard(const EnvironmentGuard&) = delete;
        EnvironmentGuard& operator=(const EnvironmentGuard&) = delete;

    private:
        const char* name_;
        std::optional<std::string> value_;
    };

    class CPUJITProbeTest : public testing::Test {
    protected:
        void SetUp() override {
            call = {};
            root_ = std::filesystem::temp_directory_path() /
                ("ormi-probe-test-" + std::to_string(ormi::platform::process_id()));
            std::filesystem::create_directories(root_);
            compiler_ = {
                root_ / "compiler path" / "driver",
                jit::CXX::gcc,
                {"PATH=path with spaces", "EMPTY=", "VALUE=a=b"}
            };
        }

        void TearDown() override {
            for (const auto& source : call.sources) {
                EXPECT_FALSE(std::filesystem::exists(source.parent_path())) << source;
            }
            std::error_code ignored;
            std::filesystem::remove_all(root_, ignored);
        }

        std::optional<std::string> probe() const {
            return jit::compatibility::probe(compiler_, std::chrono::milliseconds(1234));
        }

        std::filesystem::path root_;
        jit::Toolchain compiler_;
    };

    class CPUJITProbeArgumentsTest
        : public CPUJITProbeTest,
          public testing::WithParamInterface<jit::CXX> {};

    TEST_P(CPUJITProbeArgumentsTest, preservesSourceArgumentsEnvironmentAndTimeout) {
        compiler_.id = GetParam();
        EXPECT_FALSE(probe());

        ASSERT_EQ(call.sources.size(), 1);
        const auto& source = call.sources.front();
        const bool msvc = compiler_.id == jit::CXX::msvc ||
            compiler_.id == jit::CXX::clang_cl;
        const std::vector<std::string> expected = msvc
            ? std::vector<std::string>{"/nologo", "/std:c++20", "/Zs", source.string()}
            : std::vector<std::string>{"-std=c++20", "-fsyntax-only", source.string()};
        EXPECT_EQ(call.executable, compiler_.executable);
        EXPECT_EQ(call.arguments, expected);
        EXPECT_EQ(call.environment, compiler_.environment);
        EXPECT_EQ(call.timeout, std::chrono::milliseconds(1234));
        EXPECT_EQ(source.filename(), "probe.cpp");
        EXPECT_EQ(call.source_text,
            "#include <bit>\n"
            "#include <cmath>\n"
            "#include <cstddef>\n"
            "#include <cstdint>\n"
            "extern \"C\" int ormi_compiler_probe() {\n"
            "    return static_cast<int>(sizeof(std::size_t));\n"
            "}\n");
        EXPECT_EQ(call.source_text, jit::compatibility::source_text);
    }

    INSTANTIATE_TEST_SUITE_P(AllCompilers, CPUJITProbeArgumentsTest,
        testing::Values(jit::CXX::gcc, jit::CXX::clang, jit::CXX::clang_cl, jit::CXX::msvc),
        [](const testing::TestParamInfo<jit::CXX>& info) {
            switch (info.param) {
                case jit::CXX::gcc: return "gcc";
                case jit::CXX::clang: return "clang";
                case jit::CXX::clang_cl: return "clang_cl";
                case jit::CXX::msvc: return "msvc";
            }
            return "unknown";
        });

    TEST_F(CPUJITProbeTest, successfulProbeIgnoresCompilerOutput) {
        call.result.output = std::string(6000, 'x');
        EXPECT_FALSE(probe());
    }

    TEST_F(CPUJITProbeTest, preservesNonzeroExitDiagnostic) {
        call.result = {-3, "compiler output", Process::State::exited};
        EXPECT_EQ(probe(), "probe failed with exit code -3\ncompiler output");
    }

    TEST_F(CPUJITProbeTest, preservesTimeoutDiagnosticRegardlessOfExitCode) {
        call.result = {0, "before timeout", Process::State::timed_out};
        EXPECT_EQ(probe(), "probe timed out\nbefore timeout");
    }

    TEST_F(CPUJITProbeTest, preservesTimeoutBeforeCompilerProducesOutput) {
        call.result = {0, "", Process::State::timed_out};
        EXPECT_EQ(probe(), "probe timed out");
    }

    TEST_F(CPUJITProbeTest, preservesCrashDiagnostic) {
        call.result = {-7, "before crash", Process::State::crashed};
        EXPECT_EQ(probe(), "probe terminated abnormally with status -7\nbefore crash");
    }

    TEST_F(CPUJITProbeTest, preservesEmptyDiagnosticAndTruncationBoundary) {
        for (const auto size : {0, 4095, 4096, 4097}) {
            SCOPED_TRACE(size);
            call.result = {23, std::string(size, 'x'), Process::State::exited};
            std::string expected = "probe failed with exit code 23";
            if (size != 0) {
                expected += "\n" + std::string(size > 4096 ? 4096 : size, 'x');
                if (size > 4096) {
                    expected += "\n[compiler output truncated]";
                }
            }
            EXPECT_EQ(probe(), expected);
        }
    }

    TEST_F(CPUJITProbeTest, returnsLaunchErrorAsRejectionAndRemovesSource) {
        call.failure = Failure::launch;
        EXPECT_EQ(probe(), "cannot launch fixture compiler");
    }

    TEST_F(CPUJITProbeTest, propagatesOtherProcessErrorsAndRemovesSource) {
        call.failure = Failure::unexpected;
        EXPECT_THROW((void)probe(), std::logic_error);
    }

    TEST_F(CPUJITProbeTest, sourceSetupErrorsThrowBeforeLaunching) {
        const auto blocked = root_ / "not a directory";
        std::ofstream(blocked) << "blocked";
        const EnvironmentGuard tmpdir("TMPDIR");
        const EnvironmentGuard tmp("TMP");
        const EnvironmentGuard temp("TEMP");
        for (const auto* name : {"TMPDIR", "TMP", "TEMP"}) {
            environment(name, blocked.string().c_str());
        }

        EXPECT_THROW((void)probe(), ormi::jitError);
        EXPECT_TRUE(call.sources.empty());
    }

    TEST_F(CPUJITProbeTest, doesNotReadOrCreateDiskCache) {
        const auto blocked = root_ / "blocked cache";
        std::ofstream(blocked) << "not a cache directory";
        const EnvironmentGuard guard("ORMI_CPU_JIT_CACHE_DIR");
        environment("ORMI_CPU_JIT_CACHE_DIR", blocked.string().c_str());

        EXPECT_FALSE(probe());
        EXPECT_TRUE(std::filesystem::is_regular_file(blocked));
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(root_),
            std::filesystem::directory_iterator()), 1);
    }

    TEST_F(CPUJITProbeTest, independentCallsRunAgainWithDistinctTemporarySources) {
        EXPECT_FALSE(probe());
        call.result = {23, "", Process::State::exited};
        EXPECT_TRUE(probe());

        ASSERT_EQ(call.sources.size(), 2);
        EXPECT_NE(call.sources[0], call.sources[1]);
    }
}

namespace ormi::platform {
    Process run(
        const std::filesystem::path& executable,
        std::span<const std::string> arguments,
        std::span<const std::string> environment,
        std::chrono::milliseconds timeout
    ) {
        call.executable = executable;
        call.arguments.assign(arguments.begin(), arguments.end());
        call.environment.assign(environment.begin(), environment.end());
        call.timeout = timeout;
        const std::filesystem::path source(arguments.back());
        call.sources.push_back(source);
        std::ifstream file(source, std::ios::binary);
        if (!file) {
            throw std::logic_error("probe source is not alive during launch");
        }
        call.source_text.assign(std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>());
        if (call.failure == Failure::launch) {
            throw jitError("cannot launch fixture compiler");
        }
        if (call.failure == Failure::unexpected) {
            throw std::logic_error("unexpected fixture failure");
        }
        return call.result;
    }
}
