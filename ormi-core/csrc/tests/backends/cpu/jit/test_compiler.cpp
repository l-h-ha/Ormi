#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/jit/compiler.h"
#include "platform/platform.h"

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

    class TemporaryDirectory {
    public:
        TemporaryDirectory()
            : path_(
                std::filesystem::temp_directory_path() /
                ("ormi-compiler-test-" +
                    std::to_string(ormi::platform::process_id()))
            ) {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
            std::filesystem::create_directories(path_);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
        }

        TemporaryDirectory(const TemporaryDirectory&) = delete;
        TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

        const std::filesystem::path& path() const noexcept {
            return path_;
        }

    private:
        std::filesystem::path path_;
    };

    std::filesystem::path copy_compiler(const TemporaryDirectory& directory, std::string_view name) {
        auto executable = directory.path() / std::string(name);
#if defined(_WIN32)
        executable += ".exe";
#endif
        std::filesystem::copy_file(ORMI_PROCESS_HELPER, executable);
#if !defined(_WIN32)
        std::filesystem::permissions(
            executable,
            std::filesystem::perms::owner_exec,
            std::filesystem::perm_options::add
        );
#endif
        return executable;
    }

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

    TEST(CPUJITCompilerTest, trustsExplicitCompilerIdWithoutLaunchingIt) {
        const EnvironmentGuard compiler_guard("ORMI_CPU_JIT_CXX");
        const EnvironmentGuard id_guard("ORMI_CPU_JIT_CXX_ID");
        environment("ORMI_CPU_JIT_CXX", ORMI_PROCESS_HELPER);
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");

        const auto compiler = ormi::backends::cpu::jit::resolve();

        EXPECT_EQ(compiler.executable, std::filesystem::path(ORMI_PROCESS_HELPER));
        EXPECT_EQ(compiler.id, ormi::backends::cpu::jit::CXX::gcc);
    }

    TEST(CPUJITCompilerTest, recognizesExplicitCompilerNamesWithoutProbing) {
        const EnvironmentGuard compiler_guard("ORMI_CPU_JIT_CXX");
        const EnvironmentGuard id_guard("ORMI_CPU_JIT_CXX_ID");
        const EnvironmentGuard timeout_guard("ORMI_CPU_JIT_TIMEOUT_MS");
        const EnvironmentGuard include_guard("INCLUDE");
        const EnvironmentGuard lib_guard("LIB");
        TemporaryDirectory temporary;
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "invalid");
        environment("INCLUDE", "inherited include");
        environment("LIB", "inherited lib");
        using CXX = ormi::backends::cpu::jit::CXX;
        const std::array cases{
            std::pair{"g++", CXX::gcc},
            std::pair{"clang++", CXX::clang},
            std::pair{"clang-cl", CXX::clang_cl},
            std::pair{"cl", CXX::msvc}
        };
        for (const auto& [name, id] : cases) {
            const auto executable = copy_compiler(temporary, name);
            environment("ORMI_CPU_JIT_CXX", executable.string().c_str());

            const auto compiler = ormi::backends::cpu::jit::resolve();

            EXPECT_EQ(compiler.executable, executable);
            EXPECT_EQ(compiler.id, id);
            EXPECT_TRUE(compiler.environment.empty());
        }
    }

    TEST(CPUJITCompilerTest, explicitIdOverridesRecognizedCompilerNameWithoutProbing) {
        const EnvironmentGuard compiler_guard("ORMI_CPU_JIT_CXX");
        const EnvironmentGuard id_guard("ORMI_CPU_JIT_CXX_ID");
        const EnvironmentGuard timeout_guard("ORMI_CPU_JIT_TIMEOUT_MS");
        TemporaryDirectory temporary;
        const auto executable = copy_compiler(temporary, "clang++");
        environment("ORMI_CPU_JIT_CXX", executable.string().c_str());
        environment("ORMI_CPU_JIT_CXX_ID", "GcC");
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "invalid");

        const auto compiler = ormi::backends::cpu::jit::resolve();

        EXPECT_EQ(compiler.executable, executable);
        EXPECT_EQ(compiler.id, ormi::backends::cpu::jit::CXX::gcc);
    }

    TEST(CPUJITCompilerTest, idFilterDoesNotOverrideAutomaticCandidateSyntax) {
        const EnvironmentGuard path_guard("PATH");
        const EnvironmentGuard compiler_guard("ORMI_CPU_JIT_CXX");
        const EnvironmentGuard id_guard("ORMI_CPU_JIT_CXX_ID");
        const EnvironmentGuard timeout_guard("ORMI_CPU_JIT_TIMEOUT_MS");
        TemporaryDirectory temporary;
        (void)copy_compiler(temporary, "clang++");
        environment("PATH", temporary.path().string().c_str());
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "invalid");

        try {
            (void)ormi::backends::cpu::jit::resolve();
            FAIL() << "Expected unmatched automatic compiler syntax to be skipped.";
        } catch (const ormi::jitError& error) {
            EXPECT_NE(std::string(error.what()).find("No compatible CPU JIT compiler was found"),
                std::string::npos) << error.what();
        }
    }

    TEST(CPUJITCompilerTest, rejectsInvalidIdBeforeResolvingExplicitPath) {
        const EnvironmentGuard compiler_guard("ORMI_CPU_JIT_CXX");
        const EnvironmentGuard id_guard("ORMI_CPU_JIT_CXX_ID");
        environment("ORMI_CPU_JIT_CXX", "Z:/ormi/missing/compiler.exe");
        environment("ORMI_CPU_JIT_CXX_ID", "unknown");

        try {
            (void)ormi::backends::cpu::jit::resolve();
            FAIL() << "Expected compiler id validation before executable lookup.";
        } catch (const ormi::jitError& error) {
            EXPECT_NE(std::string(error.what()).find("ORMI_CPU_JIT_CXX_ID must be"),
                std::string::npos) << error.what();
        }
    }

    TEST(CPUJITCompilerTest, unidentifiedExplicitWrapperDoesNotFallBackToAutomaticCompiler) {
        const EnvironmentGuard compiler_guard("ORMI_CPU_JIT_CXX");
        const EnvironmentGuard id_guard("ORMI_CPU_JIT_CXX_ID");
        const EnvironmentGuard timeout_guard("ORMI_CPU_JIT_TIMEOUT_MS");
        environment("ORMI_CPU_JIT_CXX", ORMI_PROCESS_HELPER);
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TIMEOUT_MS", "5000");

        try {
            (void)ormi::backends::cpu::jit::resolve();
            FAIL() << "Expected an unidentified explicit wrapper to be rejected.";
        } catch (const ormi::jitError& error) {
            EXPECT_NE(std::string(error.what()).find("ORMI cannot identify the compiler syntax"),
                std::string::npos) << error.what();
        }
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

    TEST(CPUJITCompilerTest, rejectsIncompatibleAutomaticCompiler) {
        const EnvironmentGuard path_guard("PATH");
        const EnvironmentGuard compiler_guard("ORMI_CPU_JIT_CXX");
        const EnvironmentGuard id_guard("ORMI_CPU_JIT_CXX_ID");
        TemporaryDirectory temporary;
#if defined(_WIN32)
        const auto candidate = temporary.path() / "clang++.exe";
#else
        const auto candidate = temporary.path() / "clang++";
#endif
        std::filesystem::copy_file(
            ORMI_PROCESS_HELPER,
            candidate,
            std::filesystem::copy_options::overwrite_existing
        );
#if !defined(_WIN32)
        std::filesystem::permissions(
            candidate,
            std::filesystem::perms::owner_exec,
            std::filesystem::perm_options::add
        );
#endif
        environment("PATH", temporary.path().string().c_str());
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "clang");

        try {
            (void)ormi::backends::cpu::jit::resolve();
            FAIL() << "Expected automatic compiler compatibility rejection.";
        } catch (const ormi::jitError& error) {
            const std::string message(error.what());
            EXPECT_NE(
                message.find("No compatible CPU JIT compiler was found"),
                std::string::npos
            );
            EXPECT_NE(message.find(candidate.string()), std::string::npos);
            EXPECT_NE(message.find("probe"), std::string::npos) << message;
        }
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
