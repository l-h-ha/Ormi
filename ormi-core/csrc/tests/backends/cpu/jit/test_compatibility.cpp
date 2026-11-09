#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/jit/cache.h"
#include "backends/cpu/jit/compatibility.h"
#include "backends/cpu/jit/compiler.h"
#include "platform/platform.h"

namespace {
    namespace jit = ormi::backends::cpu::jit;
    using namespace std::chrono_literals;

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

    class CPUJITCompatibilityTest : public testing::Test {
    protected:
        void SetUp() override {
            if (const char* value = std::getenv("ORMI_CPU_JIT_CACHE_DIR")) {
                previous_cache = value;
            }
            static std::atomic_ullong next{0};
            root = std::filesystem::temp_directory_path() / (
                "ormi-compatibility-test-" + std::to_string(ormi::platform::process_id()) +
                "-" + std::to_string(next.fetch_add(1))
            );
            std::filesystem::create_directories(root);
            environment("ORMI_CPU_JIT_CACHE_DIR", (root / "cache").string().c_str());
            compiler = jit::Toolchain{
                ORMI_COMPILER_PROBE_HELPER,
                jit::CXX::gcc,
                {"ORMI_PROBE_ROOT=" + root.string()}
            };
            for (const auto* name : {"PATH", "SystemRoot"}) {
                if (const char* value = std::getenv(name)) {
                    compiler.environment.push_back(std::string(name) + "=" + value);
                }
            }
            mode("ok");
        }

        void TearDown() override {
            environment("ORMI_CPU_JIT_CACHE_DIR", previous_cache ? previous_cache->c_str() : "");
            std::error_code ignored;
            std::filesystem::remove_all(root, ignored);
        }

        void mode(std::string_view value) {
            std::ofstream file(root / "mode.txt");
            file << value;
            ASSERT_TRUE(file);
        }

        std::vector<std::filesystem::path> calls() const {
            std::vector<std::filesystem::path> result;
            std::ifstream file(root / "calls.txt");
            std::string line;
            while (std::getline(file, line)) {
                result.emplace_back(line);
            }
            return result;
        }

        bool wait_for_call() const {
            const auto deadline = std::chrono::steady_clock::now() + 5s;
            while (calls().empty()) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    return false;
                }
                std::this_thread::sleep_for(5ms);
            }
            return true;
        }

        void release() const {
            std::ofstream file(root / "release");
            file << "ready";
        }

        std::filesystem::path marker(std::string_view extension) const {
            for (const auto& entry : std::filesystem::directory_iterator(root / "cache" / "compilers")) {
                if (entry.path().extension() == extension) {
                    return entry.path();
                }
            }
            return {};
        }

        std::optional<std::string> check(std::chrono::milliseconds timeout = 5s) const {
            return jit::compatibility::check(compiler, timeout);
        }

        void expect_sources_removed() const {
            for (const auto& source : calls()) {
                EXPECT_FALSE(std::filesystem::exists(source.parent_path())) << source;
            }
        }

        std::optional<std::string> previous_cache;
        std::filesystem::path root;
        jit::Toolchain compiler;
    };

    TEST_F(CPUJITCompatibilityTest, reusesSuccessfulProbeWithoutLaunchingAgain) {
        EXPECT_FALSE(check());
        mode("fail");
        EXPECT_FALSE(check());
        EXPECT_EQ(calls().size(), 1);
        EXPECT_FALSE(marker(".ok").empty());
        expect_sources_removed();
    }

    TEST_F(CPUJITCompatibilityTest, reusesExistingFingerprintMarkersWithoutProbing) {
        constexpr std::string_view source =
            "#include <bit>\n"
            "#include <cmath>\n"
            "#include <cstddef>\n"
            "#include <cstdint>\n"
            "extern \"C\" int ormi_compiler_probe() {\n"
            "    return static_cast<int>(sizeof(std::size_t));\n"
            "}\n";
        std::string identity = compiler.executable.string() + "\ngcc\n" +
            std::to_string(std::filesystem::file_size(compiler.executable)) + "\n" +
            std::to_string(std::filesystem::last_write_time(compiler.executable)
                .time_since_epoch().count());
        for (const auto& variable : compiler.environment) {
            identity += "\n" + variable;
        }
        const auto fingerprint = jit::cache::key(source, identity);
        const auto directory = root / "cache" / "compilers";
        std::filesystem::create_directories(directory);
        const auto failure = directory / (fingerprint + ".fail");
        {
            std::ofstream file(failure, std::ios::binary);
            file << fingerprint << "\nprevious diagnostic";
            ASSERT_TRUE(file);
        }
        EXPECT_EQ(check(), "previous diagnostic");
        std::filesystem::remove(failure);
        {
            std::ofstream file(directory / (fingerprint + ".ok"), std::ios::binary);
            file << fingerprint;
            ASSERT_TRUE(file);
        }
        mode("fail");
        EXPECT_FALSE(check());
        EXPECT_TRUE(calls().empty());
    }

    TEST_F(CPUJITCompatibilityTest, reusesFailureAndItsDiagnosticWithoutLaunchingAgain) {
        mode("fail");
        const auto first = check();
        ASSERT_TRUE(first);
        EXPECT_NE(first->find("probe failed with exit code 23"), std::string::npos);
        EXPECT_NE(first->find("probe fixture output"), std::string::npos);
        mode("ok");
        EXPECT_EQ(check(), first);
        EXPECT_EQ(calls().size(), 1);
        EXPECT_FALSE(marker(".fail").empty());
        expect_sources_removed();
    }

    TEST_F(CPUJITCompatibilityTest, cachesTimeoutAndRemovesTemporarySource) {
        mode("wait");
        const auto first = check(300ms);
        ASSERT_TRUE(first);
        EXPECT_NE(first->find("probe timed out"), std::string::npos);
        // A timeout can kill the helper before it reaches main().
        const auto first_calls = calls();
        EXPECT_LE(first_calls.size(), 1u);
        mode("ok");
        EXPECT_EQ(check(), first);
        EXPECT_EQ(calls(), first_calls);
        expect_sources_removed();
    }

    TEST_F(CPUJITCompatibilityTest, cachesCrashAndRemovesTemporarySource) {
        mode("crash");
        const auto first = check();
        ASSERT_TRUE(first);
        EXPECT_NE(first->find("probe terminated abnormally with status"), std::string::npos);
        EXPECT_NE(first->find("probe fixture output"), std::string::npos);
        mode("ok");
        EXPECT_EQ(check(), first);
        EXPECT_EQ(calls().size(), 1);
        expect_sources_removed();
    }

    TEST_F(CPUJITCompatibilityTest, cachesProcessLaunchFailure) {
        compiler.executable = root / "missing-compiler";
        const auto first = check();
        ASSERT_TRUE(first);
        EXPECT_FALSE(first->empty());
        EXPECT_EQ(check(), first);
        EXPECT_TRUE(calls().empty());
        EXPECT_FALSE(marker(".fail").empty());
    }

    TEST_F(CPUJITCompatibilityTest, truncatesCompilerOutputAndReusesExactDiagnostic) {
        mode("noisy");
        const auto first = check();
        ASSERT_TRUE(first);
        EXPECT_NE(first->find("[compiler output truncated]"), std::string::npos);
        EXPECT_LT(first->size(), 4300);
        EXPECT_EQ(check(), first);
        EXPECT_EQ(calls().size(), 1);
    }

    TEST_F(CPUJITCompatibilityTest, invalidatesProbeWhenExplicitEnvironmentChanges) {
        EXPECT_FALSE(check());
        mode("fail");
        compiler.environment.push_back("ORMI_PROBE_REVISION=2");
        EXPECT_TRUE(check());
        EXPECT_EQ(calls().size(), 2);
        compiler.environment.pop_back();
        EXPECT_FALSE(check());
        EXPECT_EQ(calls().size(), 2);
    }

    TEST_F(CPUJITCompatibilityTest, invalidatesProbeWhenExecutableChanges) {
        const auto executable = root / compiler.executable.filename();
        std::filesystem::copy_file(compiler.executable, executable);
        compiler.executable = executable;
        EXPECT_FALSE(check());
        mode("fail");
        const auto modified = std::filesystem::last_write_time(executable);
        std::filesystem::last_write_time(executable, modified + 2s);
        EXPECT_TRUE(check());
        EXPECT_EQ(calls().size(), 2);
    }

    TEST_F(CPUJITCompatibilityTest, usesBothCompilerArgumentFamilies) {
        const std::array ids{jit::CXX::gcc, jit::CXX::clang, jit::CXX::msvc, jit::CXX::clang_cl};
        for (const auto id : ids) {
            compiler.id = id;
            EXPECT_FALSE(check()) << jit::to_string(id);
        }
        EXPECT_EQ(calls().size(), ids.size());
    }

    TEST_F(CPUJITCompatibilityTest, ignoresMalformedSuccessRecordAndReprobes) {
        EXPECT_FALSE(check());
        const auto path = marker(".ok");
        ASSERT_FALSE(path.empty());
        {
            std::ofstream file(path, std::ios::trunc);
            file << std::string(32, 'x');
        }
        mode("fail");
        EXPECT_TRUE(check());
        EXPECT_EQ(calls().size(), 2);
        EXPECT_FALSE(std::filesystem::exists(path));
    }

    TEST_F(CPUJITCompatibilityTest, ignoresMalformedFailureRecordAndReprobes) {
        mode("fail");
        EXPECT_TRUE(check());
        const auto path = marker(".fail");
        ASSERT_FALSE(path.empty());
        {
            std::ofstream file(path, std::ios::trunc);
            file << "incomplete";
        }
        mode("ok");
        EXPECT_FALSE(check());
        EXPECT_EQ(calls().size(), 2);
        EXPECT_FALSE(std::filesystem::exists(path));
    }

    TEST_F(CPUJITCompatibilityTest, concurrentRequestsLaunchOnlyOneProbe) {
        mode("wait");
        std::promise<void> start;
        const auto ready = start.get_future().share();
        std::array<std::future<std::optional<std::string>>, 4> requests;
        for (auto& request : requests) {
            request = std::async(std::launch::async, [this, ready] {
                ready.wait();
                return check();
            });
        }
        start.set_value();
        const bool launched = wait_for_call();
        release();
        EXPECT_TRUE(launched);
        for (auto& request : requests) {
            EXPECT_FALSE(request.get());
        }
        EXPECT_EQ(calls().size(), 1);
        expect_sources_removed();
    }

    TEST_F(CPUJITCompatibilityTest, clearWaitsForActiveProbeAndThenInvalidatesItsResult) {
        mode("wait");
        auto active = std::async(std::launch::async, [this] { return check(); });
        const bool launched = wait_for_call();
        std::promise<void> entering_clear;
        auto clearing = std::async(std::launch::async, [&entering_clear] {
            entering_clear.set_value();
            jit::compatibility::clear();
        });
        entering_clear.get_future().wait();
        const auto state = clearing.wait_for(100ms);
        release();
        EXPECT_TRUE(launched);
        EXPECT_EQ(state, std::future_status::timeout);
        EXPECT_FALSE(active.get());
        clearing.get();
        EXPECT_TRUE(marker(".ok").empty());
        mode("fail");
        EXPECT_TRUE(check());
        EXPECT_EQ(calls().size(), 2);
        expect_sources_removed();
    }

    TEST_F(CPUJITCompatibilityTest, cachePublicationErrorsRemainExceptions) {
        mode("wait");
        auto active = std::async(std::launch::async, [this] { return check(); });
        const bool launched = wait_for_call();
        for (const auto& entry : std::filesystem::directory_iterator(root / "cache" / "compilers")) {
            if (entry.path().extension() == ".lock" && entry.path().filename() != ".maintenance.lock") {
                auto blocked = entry.path();
                blocked.replace_extension(".ok");
                std::filesystem::create_directory(blocked);
            }
        }
        release();
        EXPECT_TRUE(launched);
        EXPECT_THROW(active.get(), ormi::jitError);
        EXPECT_EQ(calls().size(), 1);
        expect_sources_removed();
    }
}
