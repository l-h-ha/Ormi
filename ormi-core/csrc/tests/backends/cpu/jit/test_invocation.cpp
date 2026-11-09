#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/math.h"
#include "backends/cpu/jit/target.h"
#include "platform/platform.h"

namespace {
    namespace jit = ormi::backends::cpu::jit;
    using Process = ormi::platform::Process;

    struct Invocation {
        int calls = 0;
        std::filesystem::path executable;
        std::vector<std::string> arguments;
        std::vector<std::string> environment;
        std::chrono::milliseconds timeout{};
        Process result{0, "compiler diagnostics\n", Process::State::exited};
    };

    Invocation invocation;

    void timeout(const char* value) {
#if defined(_WIN32)
        _putenv_s("ORMI_CPU_JIT_TIMEOUT_MS", value ? value : "");
#else
        if (value) {
            setenv("ORMI_CPU_JIT_TIMEOUT_MS", value, 1);
        } else {
            unsetenv("ORMI_CPU_JIT_TIMEOUT_MS");
        }
#endif
    }

    class CPUJITInvocationTest : public testing::Test {
    protected:
        void SetUp() override {
            if (const char* value = std::getenv("ORMI_CPU_JIT_TIMEOUT_MS")) {
                previous_timeout_ = value;
            }
            timeout(nullptr);
            invocation = {};
            root_ = std::filesystem::temp_directory_path() /
                ("ormi invocation " + std::to_string(ormi::platform::process_id()));
            std::filesystem::create_directories(root_);
            source_ = root_ / "kernel source.cpp";
            output_ = root_ / "kernel library.dll";
            object_ = root_ / "kernel source.obj";
        }

        void TearDown() override {
            timeout(previous_timeout_ ? previous_timeout_->c_str() : nullptr);
            std::error_code ignored;
            std::filesystem::remove_all(root_, ignored);
        }

        void write(const std::filesystem::path& path) {
            std::ofstream file(path);
            file << "fixture";
            ASSERT_TRUE(file.good());
        }

        jit::Toolchain compiler_{
            std::filesystem::path("compiler directory") / "compiler runner",
            jit::CXX::gcc,
            {"INCLUDE=include directory", "LIB=library directory", "EMPTY="}
        };
        std::filesystem::path root_;
        std::filesystem::path source_;
        std::filesystem::path output_;
        std::filesystem::path object_;

    private:
        std::optional<std::string> previous_timeout_;
    };

    class CPUJITInvocationArgumentsTest
        : public CPUJITInvocationTest,
          public testing::WithParamInterface<jit::CXX> {};

    TEST_P(CPUJITInvocationArgumentsTest, preservesCompleteArgumentOrder) {
        compiler_.id = GetParam();
        const bool msvc = compiler_.id == jit::CXX::msvc ||
            compiler_.id == jit::CXX::clang_cl;
        const std::array targets{
            jit::Target{},
            jit::Target{.avx2 = true},
            jit::Target{.avx2 = true, .f16c = true}
        };
        const int functions = 0;
        const std::array math{
            jit::Math{},
            jit::Math{nullptr, jit::MathABI::direct},
            jit::Math{&functions, jit::MathABI::buffered},
            jit::Math{&functions, jit::MathABI::direct}
        };
        write(output_);

        for (const auto target : targets) {
            for (const auto functions : math) {
                SCOPED_TRACE(jit::to_string(target));
                SCOPED_TRACE(static_cast<int>(functions.abi));
                std::vector<std::string> expected = msvc
                    ? std::vector<std::string>{"/nologo", "/std:c++20", "/O2"}
                    : std::vector<std::string>{"-std=c++20", "-O3", "-DNDEBUG"};
                const std::string define = msvc ? "/D" : "-D";
                if (functions.functions) {
                    expected.push_back(define + "ORMI_USE_VECMATH=1");
                    if (functions.abi == jit::MathABI::direct) {
                        expected.push_back(define + "ORMI_USE_DIRECT_VECMATH=1");
                    }
                }
                if (target.avx2) {
                    expected.emplace_back(msvc ? "/arch:AVX2" : "-mavx2");
                    expected.push_back(define + "ORMI_USE_AVX2=1");
                }
                if (target.f16c) {
                    if (!msvc) {
                        expected.emplace_back("-mf16c");
                    }
                    expected.push_back(define + "ORMI_USE_F16C=1");
                }
                if (msvc) {
                    const std::vector<std::string> tail{
                        "/LD", "/EHsc", "/Fo:" + object_.string(),
                        source_.string(), "/Fe:" + output_.string(),
                        "/link", "/NOIMPLIB"
                    };
                    expected.insert(expected.end(), tail.begin(), tail.end());
                } else {
                    const auto shared = ormi::platform::shared_flags();
                    expected.insert(expected.end(), shared.begin(), shared.end());
                    expected.emplace_back("-o");
                    expected.push_back(output_.string());
                    expected.push_back(source_.string());
                }

                const auto previous = invocation.calls;
                jit::build(compiler_, target, functions, source_, output_);

                EXPECT_EQ(invocation.calls, previous + 1);
                EXPECT_EQ(invocation.executable, compiler_.executable);
                EXPECT_EQ(invocation.arguments, expected);
                EXPECT_EQ(invocation.environment, compiler_.environment);
                EXPECT_EQ(invocation.timeout, std::chrono::milliseconds(120000));
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(AllCompilers, CPUJITInvocationArgumentsTest,
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

    TEST_F(CPUJITInvocationTest, passesArgumentsAndEnvironmentWithoutRewriting) {
        const std::vector<std::string> arguments{"", "a b", "quote\"here", "tail\\"};
        const auto result = jit::run(compiler_, arguments);

        EXPECT_EQ(invocation.arguments, arguments);
        EXPECT_EQ(invocation.environment, compiler_.environment);
        EXPECT_EQ(result.output, invocation.result.output);
    }

    TEST_F(CPUJITInvocationTest, readsTimeoutOnEachInvocation) {
        (void)jit::run(compiler_, {});
        EXPECT_EQ(invocation.timeout, std::chrono::milliseconds(120000));
        timeout("");
        (void)jit::run(compiler_, {});
        EXPECT_EQ(invocation.timeout, std::chrono::milliseconds(120000));
        timeout("0");
        (void)jit::run(compiler_, {});
        EXPECT_EQ(invocation.timeout, (std::chrono::milliseconds::max)());
        timeout("0017");
        (void)jit::run(compiler_, {});
        EXPECT_EQ(invocation.timeout, std::chrono::milliseconds(17));
    }

    TEST_F(CPUJITInvocationTest, rejectsInvalidTimeoutsBeforeLaunching) {
        for (const auto value : {"-1", "+1", "1 ", " 1", "abc", "18446744073709551616"}) {
            SCOPED_TRACE(value);
            timeout(value);
            try {
                (void)jit::run(compiler_, {});
                FAIL() << "Expected timeout validation to fail.";
            } catch (const ormi::jitError& error) {
                EXPECT_STREQ(error.what(),
                    "ORMI_CPU_JIT_TIMEOUT_MS must be an unsigned decimal millisecond count.");
            }
        }
        EXPECT_EQ(invocation.calls, 0);
    }

    TEST_F(CPUJITInvocationTest, rejectsUnrepresentableTimeoutBeforeLaunching) {
        timeout("18446744073709551615");
        try {
            (void)jit::run(compiler_, {});
            FAIL() << "Expected timeout overflow to fail.";
        } catch (const ormi::jitError& error) {
            EXPECT_STREQ(error.what(), "ORMI_CPU_JIT_TIMEOUT_MS is too large.");
        }
        EXPECT_EQ(invocation.calls, 0);
    }

    TEST_F(CPUJITInvocationTest, returnsOrdinaryNonzeroExitForCallerToInterpret) {
        invocation.result.code = 23;
        const auto result = jit::run(compiler_, {});

        EXPECT_EQ(result.code, 23);
        EXPECT_EQ(result.state, Process::State::exited);
        EXPECT_EQ(result.output, invocation.result.output);
    }

    TEST_F(CPUJITInvocationTest, preservesTimeoutDiagnostic) {
        timeout("17");
        invocation.result.state = Process::State::timed_out;
        try {
            (void)jit::run(compiler_, {});
            FAIL() << "Expected compiler timeout.";
        } catch (const ormi::jitError& error) {
            EXPECT_EQ(error.what(), "CPU JIT compiler timed out after 17 ms.\nCompiler: " +
                compiler_.executable.string() + "\n" + invocation.result.output);
        }
    }

    TEST_F(CPUJITInvocationTest, preservesCrashDiagnostic) {
        invocation.result = {-7, "crash output", Process::State::crashed};
        try {
            (void)jit::run(compiler_, {});
            FAIL() << "Expected compiler crash.";
        } catch (const ormi::jitError& error) {
            EXPECT_EQ(error.what(),
                "CPU JIT compiler terminated abnormally with status -7.\nCompiler: " +
                    compiler_.executable.string() + "\ncrash output");
        }
    }

    TEST_F(CPUJITInvocationTest, removesObjectAfterSuccessfulBuild) {
        write(output_);
        write(object_);

        EXPECT_NO_THROW(jit::build(compiler_, {}, {}, source_, output_));
        EXPECT_FALSE(std::filesystem::exists(object_));
    }

    TEST_F(CPUJITInvocationTest, rejectsFailedBuildEvenWithAnOutputFile) {
        write(output_);
        write(object_);
        invocation.result.code = 23;
        try {
            jit::build(compiler_, {}, {}, source_, output_);
            FAIL() << "Expected nonzero compiler exit to fail the build.";
        } catch (const ormi::jitError& error) {
            EXPECT_EQ(error.what(), "CPU JIT compiler failed with exit code 23.\nCompiler: " +
                compiler_.executable.string() + "\nTarget: baseline\n" + invocation.result.output);
        }
        EXPECT_FALSE(std::filesystem::exists(object_));
    }

    TEST_F(CPUJITInvocationTest, rejectsSuccessfulExitWithoutAnOutputFile) {
        write(object_);
        EXPECT_THROW(jit::build(compiler_, {}, {}, source_, output_), ormi::jitError);
        EXPECT_FALSE(std::filesystem::exists(object_));
    }

    TEST_F(CPUJITInvocationTest, rejectsDirectoryAsOutputFile) {
        std::filesystem::create_directory(output_);
        EXPECT_THROW(jit::build(compiler_, {}, {}, source_, output_), ormi::jitError);
    }

    TEST_F(CPUJITInvocationTest, rejectsInvalidTargetBeforeLaunching) {
        const jit::Target target{.f16c = true};
        EXPECT_THROW(jit::build(compiler_, target, {}, source_, output_), ormi::jitError);
        EXPECT_EQ(invocation.calls, 0);
    }

    TEST_F(CPUJITInvocationTest, leavesObjectUntouchedWhenProcessDoesNotExitNormally) {
        write(object_);
        invocation.result.state = Process::State::timed_out;
        EXPECT_THROW(jit::build(compiler_, {}, {}, source_, output_), ormi::jitError);
        EXPECT_TRUE(std::filesystem::exists(object_));
    }
}

namespace ormi::platform {
    Process run(
        const std::filesystem::path& executable,
        std::span<const std::string> arguments,
        std::span<const std::string> environment,
        std::chrono::milliseconds timeout
    ) {
        ++invocation.calls;
        invocation.executable = executable;
        invocation.arguments.assign(arguments.begin(), arguments.end());
        invocation.environment.assign(environment.begin(), environment.end());
        invocation.timeout = timeout;
        return invocation.result;
    }
}
