#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/jit/compiler.h"

namespace {
    void environment(const char* name, const char* value) {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        if (*value) {
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
            environment(name_, value_ ? value_->c_str() : "");
        }

        EnvironmentGuard(const EnvironmentGuard&) = delete;
        EnvironmentGuard& operator=(const EnvironmentGuard&) = delete;

    private:
        const char* name_;
        std::optional<std::string> value_;
    };

    TEST(CPUJITCompilerTest, autoDetectsBuilderCompiler) {
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");

        const auto compiler = ormi::backends::cpu::jit::resolve();

        EXPECT_FALSE(compiler.executable.empty());
        EXPECT_FALSE(ormi::backends::cpu::jit::to_string(compiler.id).empty());
    }

    TEST(CPUJITCompilerTest, acceptsCaseInsensitiveExplicitId) {
        const auto detected = ormi::backends::cpu::jit::resolve();
        environment("ORMI_CPU_JIT_CXX", detected.executable.string().c_str());
        std::string id(ormi::backends::cpu::jit::to_string(detected.id));
        std::transform(id.begin(), id.end(), id.begin(), [](unsigned char character) {
            return static_cast<char>(std::toupper(character));
        });
        environment("ORMI_CPU_JIT_CXX_ID", id.c_str());

        const auto explicit_compiler = ormi::backends::cpu::jit::resolve();

        EXPECT_EQ(explicit_compiler.executable, detected.executable);
        EXPECT_EQ(explicit_compiler.id, detected.id);
    }

    TEST(CPUJITCompilerTest, idWithoutPathFiltersAutomaticDiscovery) {
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        const auto detected = ormi::backends::cpu::jit::resolve();
        environment(
            "ORMI_CPU_JIT_CXX_ID",
            std::string(ormi::backends::cpu::jit::to_string(detected.id)).c_str()
        );

        const auto filtered = ormi::backends::cpu::jit::resolve();

        EXPECT_EQ(filtered.id, detected.id);
    }

    TEST(CPUJITCompilerTest, rejectsInvalidExplicitCompiler) {
        environment("ORMI_CPU_JIT_CXX", "Z:/ormi/missing/compiler.exe");
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");

        EXPECT_THROW((void)ormi::backends::cpu::jit::resolve(), ormi::jitError);
    }

    TEST(CPUJITCompilerTest, timesOutCompilerProcess) {
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "100");
        const ormi::backends::cpu::jit::Toolchain compiler{
            std::filesystem::path(ORMI_PROCESS_HELPER),
            ormi::backends::cpu::jit::CXX::gcc,
            {}
        };

        try {
            (void)ormi::backends::cpu::jit::run(
                compiler,
                std::vector<std::string>{"sleep"}
            );
            FAIL() << "Expected the compiler process to time out.";
        } catch (const ormi::jitError& error) {
            EXPECT_NE(
                std::string(error.what()).find("timed out after 100 ms"),
                std::string::npos
            );
        }
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "");
    }

    TEST(CPUJITCompilerTest, reportsCompilerCrash) {
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "5000");
        const ormi::backends::cpu::jit::Toolchain compiler{
            std::filesystem::path(ORMI_PROCESS_HELPER),
            ormi::backends::cpu::jit::CXX::gcc,
            {}
        };

        try {
            (void)ormi::backends::cpu::jit::run(
                compiler,
                std::vector<std::string>{"crash"}
            );
            FAIL() << "Expected the compiler process to crash.";
        } catch (const ormi::jitError& error) {
            EXPECT_NE(
                std::string(error.what()).find("terminated abnormally"),
                std::string::npos
            );
        }
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "");
    }

    TEST(CPUJITCompilerTest, rejectsInvalidTimeoutBeforeSpawn) {
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "two minutes");
        const ormi::backends::cpu::jit::Toolchain compiler{
            std::filesystem::path(ORMI_PROCESS_HELPER),
            ormi::backends::cpu::jit::CXX::gcc,
            {}
        };

        EXPECT_THROW(
            (void)ormi::backends::cpu::jit::run(
                compiler,
                std::vector<std::string>{"normal"}
            ),
            ormi::jitError
        );
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "");
    }

#if defined(_WIN32)
    TEST(CPUJITCompilerTest, discoversConfiguredMSVCOutsideDeveloperShell) {
        const EnvironmentGuard include("INCLUDE");
        const EnvironmentGuard lib("LIB");
        environment("INCLUDE", "");
        environment("LIB", "");
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "msvc");

        try {
            const auto compiler = ormi::backends::cpu::jit::resolve();

            EXPECT_EQ(compiler.id, ormi::backends::cpu::jit::CXX::msvc);
            EXPECT_EQ(compiler.executable.filename(), "cl.exe");
            EXPECT_FALSE(compiler.environment.empty());

            std::unordered_set<std::string> keys;
            bool compiler_on_path = false;
            for (const auto& variable : compiler.environment) {
                const auto separator = variable.find('=');
                auto key = variable.substr(0, separator);
                std::transform(
                    key.begin(),
                    key.end(),
                    key.begin(),
                    [](unsigned char character) {
                        return static_cast<char>(std::tolower(character));
                    }
                );
                const bool is_path = key == "path";
                EXPECT_TRUE(keys.insert(std::move(key)).second);

                if (is_path) {
                    auto value = variable.substr(separator + 1);
                    auto compiler_directory = compiler.executable.parent_path().string();
                    std::transform(
                        value.begin(),
                        value.end(),
                        value.begin(),
                        [](unsigned char character) {
                            return static_cast<char>(std::tolower(character));
                        }
                    );
                    std::transform(
                        compiler_directory.begin(),
                        compiler_directory.end(),
                        compiler_directory.begin(),
                        [](unsigned char character) {
                            return static_cast<char>(std::tolower(character));
                        }
                    );
                    compiler_on_path |=
                        value.find(compiler_directory) != std::string::npos;
                }
            }
            EXPECT_TRUE(compiler_on_path);
        } catch (const ormi::jitError& error) {
            GTEST_SKIP() << "Configured MSVC is unavailable: " << error.what();
        }
    }
#endif
}
