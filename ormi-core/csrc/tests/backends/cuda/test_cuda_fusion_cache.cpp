#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cuda/fusion/cache.h"
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

    std::optional<std::string> environment(const char* name) {
        const auto* value = std::getenv(name);
        if (!value) {
            return std::nullopt;
        }
        return value;
    }

    void restore(
        const char* name,
        const std::optional<std::string>& value
    ) {
        environment(name, value ? value->c_str() : "");
    }

    class Directory {
    public:
        Directory()
            : path(
                std::filesystem::temp_directory_path() /
                (
                    "ormi-cuda-jit-cache-test-" +
                    std::to_string(ormi::platform::process_id())
                )
            ),
            previous_directory(
                environment("ORMI_CUDA_JIT_CACHE_DIR")
            ),
            previous_limit(
                environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES")
            )
        {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
            environment(
                "ORMI_CUDA_JIT_CACHE_DIR",
                path.string().c_str()
            );
        }

        ~Directory() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
            restore(
                "ORMI_CUDA_JIT_CACHE_DIR",
                previous_directory
            );
            restore(
                "ORMI_CUDA_JIT_CACHE_MAX_BYTES",
                previous_limit
            );
        }

        std::filesystem::path source(
            std::string_view generated,
            std::string_view context = {}
        ) const {
            return path /
                (
                    ormi::backends::cuda::fusion::cache::key(
                        generated,
                        context
                    ) +
                    ".cu"
                );
        }

        std::filesystem::path path;

    private:
        std::optional<std::string> previous_directory;
        std::optional<std::string> previous_limit;
    };

    std::string build() {
        ++builds;
        return "ptx";
    }

    TEST(CUDAFusionCacheTest, keysIncludeSourceAndTargetContext) {
        const auto first =
            ormi::backends::cuda::fusion::cache::key(
                "source",
                "compute_89\nnvrtc_12.4"
            );

        EXPECT_EQ(
            first,
            ormi::backends::cuda::fusion::cache::key(
                "source",
                "compute_89\nnvrtc_12.4"
            )
        );
        EXPECT_NE(
            first,
            ormi::backends::cuda::fusion::cache::key(
                "source",
                "compute_86\nnvrtc_12.4"
            )
        );
        EXPECT_NE(
            first,
            ormi::backends::cuda::fusion::cache::key(
                "other",
                "compute_89\nnvrtc_12.4"
            )
        );
        EXPECT_EQ(first.size(), 32u);
    }

    TEST(CUDAFusionCacheTest, reusesPublishedPTX) {
        Directory directory;
        builds = 0;

        const auto first =
            ormi::backends::cuda::fusion::cache::get(
                "source",
                "target",
                build
            );
        const auto second =
            ormi::backends::cuda::fusion::cache::get(
                "source",
                "target",
                build
            );

        EXPECT_EQ(builds, 1);
        EXPECT_FALSE(first.cached);
        EXPECT_TRUE(second.cached);
        EXPECT_EQ(first.ptx, "ptx");
        EXPECT_EQ(second.ptx, "ptx");
    }

    TEST(CUDAFusionCacheTest, serializesConcurrentMisses) {
        Directory directory;
        builds = 0;
        const auto slow_build = [] {
            ++builds;
            std::this_thread::sleep_for(
                std::chrono::milliseconds(100)
            );
            return std::string("ptx");
        };

        auto first = std::async(std::launch::async, [&] {
            return ormi::backends::cuda::fusion::cache::get(
                "concurrent",
                "target",
                slow_build
            );
        });
        auto second = std::async(std::launch::async, [&] {
            return ormi::backends::cuda::fusion::cache::get(
                "concurrent",
                "target",
                slow_build
            );
        });
        const auto first_artifact = first.get();
        const auto second_artifact = second.get();

        EXPECT_EQ(builds, 1);
        EXPECT_EQ(
            static_cast<int>(first_artifact.cached) +
                static_cast<int>(second_artifact.cached),
            1
        );
    }

    TEST(CUDAFusionCacheTest, reusesPTXAcrossProcesses) {
        Directory directory;
        const std::filesystem::path helper(
            ORMI_CUDA_CACHE_HELPER
        );
        const std::vector<std::string> build_arguments{
            directory.path.string(),
            "built"
        };
        const std::vector<std::string> load_arguments{
            directory.path.string(),
            "cached"
        };

        const auto first =
            ormi::platform::run(helper, build_arguments);
        ASSERT_EQ(first.code, 0) << first.output;

        const auto second =
            ormi::platform::run(helper, load_arguments);
        EXPECT_EQ(second.code, 0) << second.output;
    }

    TEST(CUDAFusionCacheTest, evictsLeastRecentlyUsedEntry) {
        Directory directory;
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "18");

        ormi::backends::cuda::fusion::cache::get(
            "first",
            "target",
            build
        );
        ormi::backends::cuda::fusion::cache::get(
            "second",
            "target",
            build
        );
        const auto now =
            std::filesystem::file_time_type::clock::now();
        std::filesystem::last_write_time(
            directory.source("first", "target"),
            now - std::chrono::hours(2)
        );
        std::filesystem::last_write_time(
            directory.source("second", "target"),
            now - std::chrono::hours(1)
        );

        ormi::backends::cuda::fusion::cache::get(
            "third",
            "target",
            build
        );
        const auto info =
            ormi::backends::cuda::fusion::cache::info();

        EXPECT_EQ(info.entries, 2u);
        EXPECT_LE(info.bytes, info.limit);
        EXPECT_FALSE(
            std::filesystem::exists(
                directory.source("first", "target")
            )
        );
        EXPECT_TRUE(
            std::filesystem::exists(
                directory.source("second", "target")
            )
        );
        EXPECT_TRUE(
            std::filesystem::exists(
                directory.source("third", "target")
            )
        );
    }

    TEST(CUDAFusionCacheTest, rejectsInvalidLimitBeforeBuilding) {
        Directory directory;
        environment(
            "ORMI_CUDA_JIT_CACHE_MAX_BYTES",
            "256MB"
        );
        builds = 0;

        EXPECT_THROW(
            ormi::backends::cuda::fusion::cache::get(
                "source",
                "target",
                build
            ),
            ormi::jitError
        );
        EXPECT_EQ(builds, 0);
    }

    TEST(CUDAFusionCacheTest, reportsAndClearsEntries) {
        Directory directory;
        ormi::backends::cuda::fusion::cache::get(
            "first",
            "target",
            build
        );
        ormi::backends::cuda::fusion::cache::get(
            "second",
            "target",
            build
        );

        const auto before =
            ormi::backends::cuda::fusion::cache::info();
        const auto after =
            ormi::backends::cuda::fusion::cache::clear();

        EXPECT_EQ(before.path, directory.path);
        EXPECT_EQ(before.entries, 2u);
        EXPECT_EQ(after.entries, 0u);
        EXPECT_EQ(after.bytes, 0u);
    }
}
