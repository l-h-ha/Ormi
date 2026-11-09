#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cuda/fusion/cache.h"
#include "platform/platform.h"

namespace {
    namespace cache = ormi::backends::cuda::fusion::cache;
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

        void file(std::string_view name, std::string_view contents = {}) const {
            const auto destination = path / name;
            std::filesystem::create_directories(destination.parent_path());
            std::ofstream output(destination, std::ios::binary);
            ASSERT_TRUE(output);
            output << contents;
            ASSERT_TRUE(output);
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

    TEST(CUDAFusionCacheTest, keepsProtectedEntryAboveLimit) {
        Directory directory;
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "0");
        cache::get("old", "target", build);
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "1");

        const auto artifact = cache::get("protected", "target", build);
        EXPECT_FALSE(artifact.cached);
        EXPECT_EQ(artifact.ptx, "ptx");
        const auto protected_key = cache::key("protected", "target");
        const auto old_key = cache::key("old", "target");
        EXPECT_TRUE(std::filesystem::exists(directory.path / (protected_key + ".cu")));
        EXPECT_TRUE(std::filesystem::exists(directory.path / (protected_key + ".ptx")));
        EXPECT_FALSE(std::filesystem::exists(directory.path / (old_key + ".cu")));
        EXPECT_FALSE(std::filesystem::exists(directory.path / (old_key + ".ptx")));
        EXPECT_TRUE(std::filesystem::exists(directory.path / (old_key + ".lock")));
        const auto info = cache::info();
        EXPECT_EQ(info.entries, 1u);
        EXPECT_EQ(info.bytes, 12u);
        EXPECT_EQ(info.limit, 1u);
    }

    TEST(CUDAFusionCacheTest, zeroLimitDisablesEviction) {
        Directory directory;
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "0");
        builds = 0;
        for (const auto* source : {"first", "second", "third"}) {
            EXPECT_FALSE(cache::get(source, "target", build).cached);
        }
        for (const auto* source : {"first", "second", "third"}) {
            EXPECT_TRUE(cache::get(source, "target", build).cached);
        }
        EXPECT_EQ(builds, 3);
        const auto info = cache::info();
        EXPECT_EQ(info.entries, 3u);
        EXPECT_EQ(info.bytes, 25u);
        EXPECT_EQ(info.limit, 0u);
    }

    TEST(CUDAFusionCacheTest, cacheHitRefreshesEvictionOrder) {
        Directory directory;
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "17");
        builds = 0;
        cache::get("first", "target", build);
        cache::get("second", "target", build);
        const auto now = std::filesystem::file_time_type::clock::now();
        std::filesystem::last_write_time(directory.source("first", "target"),
            now - std::chrono::hours(2));
        std::filesystem::last_write_time(directory.source("second", "target"),
            now - std::chrono::hours(1));

        EXPECT_TRUE(cache::get("first", "target", build).cached);
        EXPECT_EQ(builds, 2);
        EXPECT_GT(std::filesystem::last_write_time(directory.source("first", "target")),
            now - std::chrono::hours(1));
        EXPECT_FALSE(cache::get("third", "target", build).cached);
        EXPECT_EQ(builds, 3);
        EXPECT_TRUE(std::filesystem::exists(directory.source("first", "target")));
        EXPECT_FALSE(std::filesystem::exists(directory.source("second", "target")));
        EXPECT_TRUE(std::filesystem::exists(directory.source("third", "target")));
        EXPECT_EQ(cache::info().bytes, 16u);
    }

    TEST(CUDAFusionCacheTest, maintenanceFailureKeepsPublishedArtifactAndAllowsRetry) {
        Directory directory;
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "0");
        cache::get("old", "target", build);
        directory.file(".maintenance.lock/keep", "block maintenance lock");
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "1");

        const auto artifact = cache::get("new", "target", build);
        EXPECT_FALSE(artifact.cached);
        EXPECT_EQ(artifact.ptx, "ptx");
        EXPECT_TRUE(std::filesystem::exists(directory.source("old", "target")));
        EXPECT_TRUE(std::filesystem::exists(directory.source("new", "target")));
        EXPECT_TRUE(cache::get("new", "target", build).cached);

        std::filesystem::remove_all(directory.path / ".maintenance.lock");
        EXPECT_FALSE(cache::get("retry", "target", build).cached);
        EXPECT_FALSE(std::filesystem::exists(directory.source("old", "target")));
        EXPECT_FALSE(std::filesystem::exists(directory.source("new", "target")));
        EXPECT_TRUE(std::filesystem::exists(directory.source("retry", "target")));
        EXPECT_EQ(cache::info().entries, 1u);
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

    TEST(CUDAFusionCacheTest, reportsCompleteEntriesAmongIncompleteFiles) {
        Directory directory;
        const std::string complete(32, 'a');
        const std::string source_only(32, 'b');
        const std::string ptx_only(32, 'c');
        const std::string source_directory(32, 'd');
        const std::string ptx_directory(32, 'e');
        directory.file(complete + ".cu", "source");
        directory.file(complete + ".ptx", "ptx");
        directory.file(source_only + ".cu", "orphan");
        directory.file(ptx_only + ".ptx", "orphan");
        std::filesystem::create_directory(directory.path / (source_directory + ".cu"));
        directory.file(source_directory + ".ptx", "ptx");
        directory.file(ptx_directory + ".cu", "source");
        std::filesystem::create_directory(directory.path / (ptx_directory + ".ptx"));

        const auto info = ormi::backends::cuda::fusion::cache::info();
        EXPECT_EQ(info.entries, 1u);
        EXPECT_EQ(info.bytes, 9u);

        const auto cleared = ormi::backends::cuda::fusion::cache::clear();
        EXPECT_EQ(cleared.entries, 0u);
        EXPECT_EQ(cleared.bytes, 0u);
        EXPECT_FALSE(std::filesystem::exists(directory.path / (complete + ".cu")));
        EXPECT_FALSE(std::filesystem::exists(directory.path / (complete + ".ptx")));
        EXPECT_TRUE(std::filesystem::exists(directory.path / (source_only + ".cu")));
        EXPECT_TRUE(std::filesystem::exists(directory.path / (ptx_only + ".ptx")));
        EXPECT_TRUE(std::filesystem::is_directory(directory.path / (source_directory + ".cu")));
        EXPECT_TRUE(std::filesystem::is_directory(directory.path / (ptx_directory + ".ptx")));
    }

    TEST(CUDAFusionCacheTest, ignoresUnrelatedStagedAndNestedFiles) {
        Directory directory;
        const std::string stem(32, 'a');
        directory.file(stem + ".cu", "source");
        directory.file(stem + ".ptx", "ptx");
        for (const auto& invalid : {std::string(32, 'z'), std::string(31, 'b'),
                std::string(33, 'c'), std::string("unrelated")}) {
            directory.file(invalid + ".cu", "ignored source");
            directory.file(invalid + ".ptx", "ignored ptx");
        }
        directory.file(stem + ".123.tmp.cu", "staged source");
        directory.file(stem + ".123.tmp.ptx", "staged ptx");
        directory.file(stem + ".lock", "lock");
        directory.file(stem + ".obj", "object");
        directory.file("nested/" + stem + ".cu", "nested source");
        directory.file("nested/" + stem + ".ptx", "nested ptx");

        const auto info = ormi::backends::cuda::fusion::cache::info();
        EXPECT_EQ(info.entries, 1u);
        EXPECT_EQ(info.bytes, 9u);
    }

    TEST(CUDAFusionCacheTest, reportsZeroByteCompleteEntries) {
        Directory directory;
        const std::string stem(32, 'a');
        directory.file(stem + ".cu");
        directory.file(stem + ".ptx");

        const auto info = ormi::backends::cuda::fusion::cache::info();
        EXPECT_EQ(info.entries, 1u);
        EXPECT_EQ(info.bytes, 0u);
        EXPECT_EQ(ormi::backends::cuda::fusion::cache::clear().entries, 0u);
    }

    TEST(CUDAFusionCacheTest, reportsUppercaseHexKeys) {
        Directory directory;
        const std::string stem = "0123456789ABCDEF0123456789ABCDEF";
        directory.file(stem + ".cu", "source");
        directory.file(stem + ".ptx", "ptx");

        const auto info = ormi::backends::cuda::fusion::cache::info();
        EXPECT_EQ(info.entries, 1u);
        EXPECT_EQ(info.bytes, 9u);
    }

    TEST(CUDAFusionCacheTest, pruningIgnoresIncompleteAndUnrelatedFiles) {
        Directory directory;
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "18");
        for (const auto* source : {"first", "second"}) {
            ormi::backends::cuda::fusion::cache::get(source, "target", build);
        }
        const auto now = std::filesystem::file_time_type::clock::now();
        std::filesystem::last_write_time(directory.source("first", "target"),
            now - std::chrono::hours(2));
        std::filesystem::last_write_time(directory.source("second", "target"),
            now - std::chrono::hours(1));
        const std::string orphan(32, 'a');
        const std::string ptx_only(32, 'b');
        directory.file(orphan + ".cu", std::string(4096, 'x'));
        directory.file(ptx_only + ".ptx", std::string(4096, 'x'));
        directory.file("unrelated.cu", std::string(4096, 'x'));
        directory.file("unrelated.ptx", std::string(4096, 'x'));
        directory.file(orphan + ".123.tmp.ptx", std::string(4096, 'x'));

        ormi::backends::cuda::fusion::cache::get("third", "target", build);
        const auto info = ormi::backends::cuda::fusion::cache::info();
        EXPECT_EQ(info.entries, 2u);
        EXPECT_EQ(info.bytes, 17u);
        EXPECT_FALSE(std::filesystem::exists(directory.source("first", "target")));
        EXPECT_TRUE(std::filesystem::exists(directory.source("second", "target")));
        EXPECT_TRUE(std::filesystem::exists(directory.source("third", "target")));
        EXPECT_TRUE(std::filesystem::exists(directory.path / (orphan + ".cu")));
        EXPECT_TRUE(std::filesystem::exists(directory.path / (ptx_only + ".ptx")));
        EXPECT_TRUE(std::filesystem::exists(directory.path / "unrelated.cu"));
    }

    TEST(CUDAFusionCacheTest, reportsMissingDirectoryAsEmpty) {
        Directory directory;
        const auto info = ormi::backends::cuda::fusion::cache::info();
        EXPECT_EQ(info.path, directory.path);
        EXPECT_EQ(info.entries, 0u);
        EXPECT_EQ(info.bytes, 0u);
        const auto cleared = ormi::backends::cuda::fusion::cache::clear();
        EXPECT_EQ(cleared.entries, 0u);
        EXPECT_EQ(cleared.bytes, 0u);
        EXPECT_FALSE(std::filesystem::exists(directory.path));
    }

    TEST(CUDAFusionCacheTest, rejectsNonDirectoryCachePath) {
        Directory directory;
        std::ofstream(directory.path, std::ios::binary) << "not a directory";
        ASSERT_TRUE(std::filesystem::is_regular_file(directory.path));

        EXPECT_THROW(ormi::backends::cuda::fusion::cache::info(), ormi::jitError);
        EXPECT_THROW(ormi::backends::cuda::fusion::cache::clear(), ormi::jitError);
        EXPECT_TRUE(std::filesystem::is_regular_file(directory.path));
    }

    TEST(CUDAFusionCacheTest, cleansStagingAfterBuildExceptionAndAllowsRetry) {
        Directory directory;
        builds = 0;
        const auto stem = cache::key("source", "target");
        const auto stale_source = stem + ".7.tmp.cu";
        const auto stale_ptx = stem + ".8.tmp.ptx";
        const auto unrelated = cache::key("other", "target") + ".9.tmp.cu";
        directory.file(stale_source, "stale source");
        directory.file(stale_ptx, "stale ptx");
        directory.file(unrelated, "unrelated");

        const auto failed_build = [&]() -> std::string {
            ++builds;
            EXPECT_FALSE(std::filesystem::exists(directory.path / stale_source));
            EXPECT_FALSE(std::filesystem::exists(directory.path / stale_ptx));
            directory.file(stale_source, "partial source");
            directory.file(stale_ptx, "partial ptx");
            throw std::runtime_error("compiler failed");
        };
        try {
            cache::get("source", "target", failed_build);
            FAIL() << "Expected the compiler exception.";
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "compiler failed");
        }

        EXPECT_EQ(builds, 1);
        EXPECT_FALSE(std::filesystem::exists(directory.path / stale_source));
        EXPECT_FALSE(std::filesystem::exists(directory.path / stale_ptx));
        EXPECT_FALSE(std::filesystem::exists(directory.path / (stem + ".cu")));
        EXPECT_FALSE(std::filesystem::exists(directory.path / (stem + ".ptx")));
        EXPECT_TRUE(std::filesystem::exists(directory.path / unrelated));
        EXPECT_EQ(cache::info().entries, 0u);

        const auto retry = cache::get("source", "target", build);
        EXPECT_FALSE(retry.cached);
        EXPECT_EQ(retry.ptx, "ptx");
        EXPECT_TRUE(cache::get("source", "target", build).cached);
        EXPECT_EQ(builds, 2);
    }

    TEST(CUDAFusionCacheTest, rejectsEmptyPTXAndAllowsRetry) {
        Directory directory;
        builds = 0;
        const auto stem = cache::key("source", "target");
        const auto empty_build = [] {
            ++builds;
            return std::string{};
        };

        EXPECT_THROW(cache::get("source", "target", empty_build), ormi::jitError);
        EXPECT_EQ(builds, 1);
        EXPECT_FALSE(std::filesystem::exists(directory.path / (stem + ".cu")));
        EXPECT_FALSE(std::filesystem::exists(directory.path / (stem + ".ptx")));
        EXPECT_EQ(cache::info().entries, 0u);
        const auto retry = cache::get("source", "target", build);
        EXPECT_FALSE(retry.cached);
        EXPECT_EQ(retry.ptx, "ptx");
        EXPECT_TRUE(cache::get("source", "target", build).cached);
        EXPECT_EQ(builds, 2);
    }

    TEST(CUDAFusionCacheTest, cachedHitDoesNotBuildOrValidateMissOnlyLimit) {
        Directory directory;
        cache::get("source", "target", build);
        environment("ORMI_CUDA_JIT_CACHE_MAX_BYTES", "invalid");
        const auto unexpected_build = []() -> std::string {
            ADD_FAILURE() << "A persistent cache hit must not build again.";
            throw std::runtime_error("unexpected compiler call");
        };

        const auto cached = cache::get("source", "target", unexpected_build);
        EXPECT_TRUE(cached.cached);
        EXPECT_EQ(cached.ptx, "ptx");
    }

    TEST(CUDAFusionCacheTest, cleansStagingAfterWriteFailuresAndAllowsRetry) {
        for (const std::string extension : {".cu", ".ptx"}) {
            SCOPED_TRACE(extension);
            Directory directory;
            builds = 0;
            const auto stem = cache::key("source", "target");
            const auto staged = stem + "." +
                std::to_string(ormi::platform::process_id()) + ".tmp";
            const auto blocked = directory.path / (staged + extension);
            const auto failed_build = [&] {
                std::filesystem::create_directories(blocked);
                std::ofstream(blocked / "keep") << "prevent directory removal";
                return build();
            };

            EXPECT_THROW(cache::get("source", "target", failed_build), ormi::jitError);
            EXPECT_EQ(builds, 1);
            EXPECT_FALSE(std::filesystem::is_regular_file(directory.path / (staged + ".cu")));
            EXPECT_FALSE(std::filesystem::is_regular_file(directory.path / (staged + ".ptx")));
            EXPECT_FALSE(std::filesystem::exists(directory.path / (stem + ".cu")));
            EXPECT_FALSE(std::filesystem::exists(directory.path / (stem + ".ptx")));
            EXPECT_EQ(cache::info().entries, 0u);

            std::filesystem::remove_all(blocked);
            const auto retry = cache::get("source", "target", build);
            EXPECT_FALSE(retry.cached);
            EXPECT_EQ(retry.ptx, "ptx");
            EXPECT_TRUE(cache::get("source", "target", build).cached);
            EXPECT_EQ(builds, 2);
            EXPECT_EQ(cache::info().bytes, 9u);
        }
    }

    TEST(CUDAFusionCacheTest, cleansStagingAfterReplacementFailuresAndAllowsRetry) {
        for (const std::string extension : {".ptx", ".cu"}) {
            SCOPED_TRACE(extension);
            Directory directory;
            builds = 0;
            const auto stem = cache::key("source", "target");
            const auto staged = stem + "." +
                std::to_string(ormi::platform::process_id()) + ".tmp";
            const auto blocked = directory.path / (stem + extension);
            const auto failed_build = [&] {
                std::filesystem::create_directories(blocked);
                std::ofstream(blocked / "keep") << "prevent directory removal";
                return build();
            };

            EXPECT_THROW(cache::get("source", "target", failed_build), ormi::jitError);
            EXPECT_EQ(builds, 1);
            EXPECT_FALSE(std::filesystem::exists(directory.path / (staged + ".cu")));
            EXPECT_FALSE(std::filesystem::exists(directory.path / (staged + ".ptx")));
            EXPECT_FALSE(std::filesystem::is_regular_file(directory.path / (stem + ".cu")));
            EXPECT_EQ(std::filesystem::is_regular_file(directory.path / (stem + ".ptx")),
                extension == ".cu");
            EXPECT_EQ(cache::info().entries, 0u);

            std::filesystem::remove_all(blocked);
            const auto retry = cache::get("source", "target", build);
            EXPECT_FALSE(retry.cached);
            EXPECT_EQ(retry.ptx, "ptx");
            EXPECT_TRUE(cache::get("source", "target", build).cached);
            EXPECT_EQ(builds, 2);
            EXPECT_EQ(cache::info().bytes, 9u);
        }
    }
}
