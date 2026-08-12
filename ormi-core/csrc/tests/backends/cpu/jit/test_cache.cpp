#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/core/jit.h>

#include "backends/cpu/jit/cache.h"
#include "platform/platform.h"

namespace {
    std::atomic<int> builds{0};

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

    void build(
        const std::filesystem::path&,
        const std::filesystem::path& output
    ) {
        ++builds;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::ofstream library(output, std::ios::binary | std::ios::trunc);
        library << "artifact";
    }

    class Directory {
    public:
        Directory()
            : path(
                std::filesystem::temp_directory_path() /
                ("ormi-jit-cache-test-" +
                    std::to_string(ormi::platform::process_id()))
            )
        {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }

        ~Directory() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
            environment("ORMI_CPU_JIT_CACHE_DIR", "");
            environment("ORMI_CPU_JIT_CACHE_MAX_BYTES", "");
        }

        std::filesystem::path path;
    };

    std::filesystem::path source(const Directory& cache, std::string_view generated) {
        return cache.path /
            (ormi::backends::cpu::jit::cache::key(generated) + ".cpp");
    }

    std::filesystem::path library(const Directory& cache, std::string_view generated) {
        return cache.path /
            (
                ormi::backends::cpu::jit::cache::key(generated) +
                std::string(ormi::platform::shared_extension())
            );
    }

    TEST(CPUJITCacheTest, keysIncludeGeneratedSource) {
        const auto first = ormi::backends::cpu::jit::cache::key("first");

        EXPECT_EQ(first, ormi::backends::cpu::jit::cache::key("first"));
        EXPECT_NE(first, ormi::backends::cpu::jit::cache::key("second"));
        EXPECT_EQ(first.size(), 32);
    }

    TEST(CPUJITCacheTest, keysIncludeTargetContext) {
        const auto baseline = ormi::backends::cpu::jit::cache::key(
            "generated",
            "baseline"
        );
        const auto avx2 = ormi::backends::cpu::jit::cache::key(
            "generated",
            "avx2"
        );

        EXPECT_NE(baseline, avx2);
    }

    TEST(CPUJITCacheTest, findsCompleteArtifactWithoutBuilding) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        builds = 0;

        EXPECT_FALSE(
            ormi::backends::cpu::jit::cache::find("generated", "direct")
        );
        const auto built = ormi::backends::cpu::jit::cache::get(
            "generated",
            "direct",
            build
        );
        auto found = ormi::backends::cpu::jit::cache::find(
            "generated",
            "direct"
        );

        ASSERT_TRUE(found.has_value());
        EXPECT_EQ(builds, 1);
        EXPECT_EQ(found->library, built.library);
        EXPECT_TRUE(found->cached);
    }

    TEST(CPUJITCacheTest, serializesConcurrentMisses) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        builds = 0;

        auto first = std::async(std::launch::async, [] {
            return ormi::backends::cpu::jit::cache::get("concurrent", build);
        });
        auto second = std::async(std::launch::async, [] {
            return ormi::backends::cpu::jit::cache::get("concurrent", build);
        });
        const auto first_artifact = first.get();
        const auto second_artifact = second.get();

        EXPECT_EQ(builds, 1);
        EXPECT_EQ(first_artifact.library, second_artifact.library);
        EXPECT_EQ(
            static_cast<int>(first_artifact.cached) +
                static_cast<int>(second_artifact.cached),
            1
        );
    }

    TEST(CPUJITCacheTest, serializesConcurrentProcesses) {
        Directory cache;
        const std::filesystem::path helper(ORMI_JIT_CACHE_HELPER);
        const std::vector<std::string> arguments{
            cache.path.string(),
            "compile"
        };

        auto first = std::async(std::launch::async, [&] {
            return ormi::platform::run(helper, arguments);
        });
        auto second = std::async(std::launch::async, [&] {
            return ormi::platform::run(helper, arguments);
        });

        const auto first_result = first.get();
        const auto second_result = second.get();
        EXPECT_EQ(first_result.code, 0) << first_result.output;
        EXPECT_EQ(second_result.code, 0) << second_result.output;
    }

    TEST(CPUJITCacheTest, evictsLeastRecentlyUsedEntryAfterPublishing) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        environment("ORMI_CPU_JIT_CACHE_MAX_BYTES", "28");
        builds = 0;

        ormi::backends::cpu::jit::cache::get("first", build);
        ormi::backends::cpu::jit::cache::get("second", build);
        const auto now = std::filesystem::file_time_type::clock::now();
        std::filesystem::last_write_time(source(cache, "first"), now - std::chrono::hours(2));
        std::filesystem::last_write_time(source(cache, "second"), now - std::chrono::hours(1));

        EXPECT_TRUE(ormi::backends::cpu::jit::cache::get("second", build).cached);
        ormi::backends::cpu::jit::cache::get("third", build);
        const auto info = ormi::core::jit::info();

        EXPECT_EQ(builds, 3);
        EXPECT_EQ(info.entries, 2);
        EXPECT_EQ(info.bytes, 27);
        EXPECT_FALSE(std::filesystem::exists(source(cache, "first")));
        EXPECT_FALSE(std::filesystem::exists(library(cache, "first")));
        EXPECT_TRUE(std::filesystem::exists(source(cache, "second")));
        EXPECT_TRUE(std::filesystem::exists(source(cache, "third")));
    }

    TEST(CPUJITCacheTest, retainsNewEntryWhenItExceedsLimit) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        environment("ORMI_CPU_JIT_CACHE_MAX_BYTES", "1");

        ormi::backends::cpu::jit::cache::get("oversized", build);
        const auto info = ormi::core::jit::info();

        EXPECT_EQ(info.entries, 1);
        EXPECT_GT(info.bytes, info.limit);
    }

    TEST(CPUJITCacheTest, zeroLimitDisablesAutomaticEviction) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        environment("ORMI_CPU_JIT_CACHE_MAX_BYTES", "0");

        ormi::backends::cpu::jit::cache::get("first", build);
        ormi::backends::cpu::jit::cache::get("second", build);
        ormi::backends::cpu::jit::cache::get("third", build);

        EXPECT_EQ(ormi::core::jit::info().entries, 3);
    }

    TEST(CPUJITCacheTest, rejectsInvalidLimitBeforeBuilding) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        environment("ORMI_CPU_JIT_CACHE_MAX_BYTES", "256MB");
        builds = 0;

        EXPECT_THROW(
            ormi::backends::cpu::jit::cache::get("invalid-limit", build),
            ormi::jitError
        );
        EXPECT_EQ(builds, 0);
        EXPECT_FALSE(std::filesystem::exists(source(cache, "invalid-limit")));
        EXPECT_FALSE(std::filesystem::exists(library(cache, "invalid-limit")));
    }

    TEST(CPUJITCacheTest, reportsAndClearsCompleteEntries) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        environment("ORMI_CPU_JIT_CACHE_MAX_BYTES", "0");

        ormi::backends::cpu::jit::cache::get("first", build);
        ormi::backends::cpu::jit::cache::get("second", build);
        const auto before = ormi::core::jit::info();
        const auto after = ormi::core::jit::clear();

        EXPECT_EQ(before.path, cache.path);
        EXPECT_EQ(before.entries, 2);
        EXPECT_EQ(before.bytes, 27);
        EXPECT_EQ(before.limit, 0);
        EXPECT_EQ(after.entries, 0);
        EXPECT_EQ(after.bytes, 0);
    }

    TEST(CPUJITCacheTest, reportsMissingDirectoryAsEmpty) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());

        const auto before = ormi::core::jit::info();
        const auto after = ormi::core::jit::clear();

        EXPECT_EQ(before.path, cache.path);
        EXPECT_EQ(before.entries, 0);
        EXPECT_EQ(before.bytes, 0);
        EXPECT_EQ(after.entries, 0);
        EXPECT_EQ(after.bytes, 0);
        EXPECT_FALSE(std::filesystem::exists(cache.path));
    }

    TEST(CPUJITCacheTest, clearWaitsForArtifactLease) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::optional<ormi::backends::cpu::jit::cache::Artifact> artifact;
        artifact.emplace(
            ormi::backends::cpu::jit::cache::get("leased", build)
        );

        auto clearing = std::async(std::launch::async, [] {
            return ormi::core::jit::clear();
        });

        EXPECT_EQ(
            clearing.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout
        );
        artifact.reset();

        EXPECT_EQ(clearing.get().entries, 0);
    }

    TEST(CPUJITCacheTest, reusesArtifactWithoutResolvingCompiler) {
        Directory cache;
        const std::filesystem::path helper(ORMI_JIT_CACHE_HELPER);
        const std::vector<std::string> compile{cache.path.string(), "compile"};
        const std::vector<std::string> load{cache.path.string(), "load"};

        const auto first = ormi::platform::run(helper, compile);
        ASSERT_EQ(first.code, 0) << first.output;

        const auto second = ormi::platform::run(helper, load);
        EXPECT_EQ(second.code, 0) << second.output;

        std::size_t sources = 0;
        std::size_t libraries = 0;
        std::size_t staged = 0;
        std::filesystem::path library;
        for (const auto& entry : std::filesystem::directory_iterator(cache.path)) {
            const auto name = entry.path().filename().string();
            sources += entry.path().extension() == ".cpp";
            if (
                entry.path().extension().string() ==
                ormi::platform::shared_extension()
            ) {
                ++libraries;
                library = entry.path();
            }
            staged += name.find(".tmp") != std::string::npos;
        }

        EXPECT_EQ(sources, 1);
        EXPECT_EQ(libraries, 1);
        EXPECT_EQ(staged, 0);

        std::ofstream corrupted(library, std::ios::binary | std::ios::trunc);
        corrupted << "invalid";
        corrupted.close();

        const auto repaired = ormi::platform::run(helper, compile);
        EXPECT_EQ(repaired.code, 0) << repaired.output;
    }
}
