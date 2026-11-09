#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <optional>
#include <stdexcept>
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

    std::string read(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(file), {});
    }

    void expect_no_staging(const Directory& cache) {
        for (const auto& entry : std::filesystem::directory_iterator(cache.path)) {
            EXPECT_EQ(entry.path().filename().string().find(".tmp"), std::string::npos)
                << entry.path();
        }
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

    TEST(CPUJITCacheTest, entryCheckDoesNotCreateMissingDirectoriesOrFiles) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());

        EXPECT_FALSE(ormi::backends::cpu::jit::cache::has_entries());
        EXPECT_FALSE(std::filesystem::exists(cache.path));

        std::filesystem::create_directories(cache.path);
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::has_entries());
        EXPECT_TRUE(std::filesystem::is_empty(cache.path));
    }

    TEST(CPUJITCacheTest, entryCheckIgnoresUnrelatedFilesAndOrphanedSources) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(cache.path / ".maintenance.lock") << "";
        std::ofstream(cache.path / "unrelated.lock") << "";
        std::ofstream(cache.path / "unrelated.dll") << "";
        std::ofstream(cache.path / "unrelated.so") << "";
        std::ofstream(source(cache, "orphan")) << "orphan";

        EXPECT_FALSE(ormi::backends::cpu::jit::cache::has_entries());
        EXPECT_EQ(read(source(cache, "orphan")), "orphan");
        EXPECT_EQ(std::distance(
            std::filesystem::directory_iterator(cache.path),
            std::filesystem::directory_iterator{}
        ), 5);
    }

    TEST(CPUJITCacheTest, entryCheckIncludesCopiedLibrariesWithoutLocks) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(library(cache, "copied")) << "library";

        EXPECT_TRUE(ormi::backends::cpu::jit::cache::has_entries());
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("copied", {}));
    }

    TEST(CPUJITCacheTest, entryCheckIncludesWriterLocksBeforeLibraryPublication) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        auto lock = source(cache, "publishing");
        lock.replace_extension(".lock");
        ormi::platform::FileLock lease(lock);

        EXPECT_TRUE(ormi::backends::cpu::jit::cache::has_entries());
        EXPECT_FALSE(std::filesystem::exists(library(cache, "publishing")));
    }

    TEST(CPUJITCacheTest, entryCheckIncludesCaseVariantsForCaseInsensitiveFilesystems) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        auto copied = library(cache, "copied");
        auto extension = std::string(ormi::platform::shared_extension());
        for (auto& character : extension) {
            character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
        }
        copied.replace_extension(extension);
        std::ofstream(copied) << "library";
        std::ofstream(source(cache, "copied")) << "copied";

        EXPECT_TRUE(ormi::backends::cpu::jit::cache::has_entries());
#if defined(_WIN32)
        EXPECT_TRUE(ormi::backends::cpu::jit::cache::find("copied", {}));
#endif
        std::filesystem::remove(copied);
        auto copied_lock = source(cache, "copied");
        copied_lock.replace_extension(".lock");
        std::filesystem::remove(copied_lock);
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::has_entries());
        auto lock = source(cache, "publishing");
        lock.replace_extension(".LoCk");
        std::ofstream(lock) << "";
        EXPECT_TRUE(ormi::backends::cpu::jit::cache::has_entries());
    }

    TEST(CPUJITCacheTest, entryCheckObservesLaterPublicationAndDirectoryChanges) {
        Directory cache;
        const auto other = cache.path / "other";
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::has_entries());

        {
            const auto built = ormi::backends::cpu::jit::cache::get("later", build);
        }
        EXPECT_TRUE(ormi::backends::cpu::jit::cache::has_entries());
        environment("ORMI_CPU_JIT_CACHE_DIR", other.string().c_str());
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::has_entries());
        EXPECT_FALSE(std::filesystem::exists(other));
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        EXPECT_TRUE(ormi::backends::cpu::jit::cache::has_entries());
    }

    TEST(CPUJITCacheTest, entryCheckReportsInvalidCacheDirectory) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::ofstream(cache.path) << "file";

        EXPECT_THROW(ormi::backends::cpu::jit::cache::has_entries(), ormi::jitError);
    }

#if !defined(_WIN32)
    TEST(CPUJITCacheTest, entryCheckAllowsKnownPathsWhenDirectoryCannotBeListed) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(source(cache, "readable")) << "readable";
        std::ofstream(library(cache, "readable")) << "library";
        auto lock = source(cache, "readable");
        lock.replace_extension(".lock");
        std::ofstream(lock) << "";
        const auto permissions = std::filesystem::status(cache.path).permissions();
        std::filesystem::permissions(cache.path, std::filesystem::perms::owner_exec);
        std::error_code error;
        const std::filesystem::directory_iterator iterator(cache.path, error);
        if (!error) {
            std::filesystem::permissions(cache.path, permissions);
            GTEST_SKIP() << "This user can list directories without read permission.";
        }

        bool possible = false;
        bool found = false;
        EXPECT_NO_THROW({
            possible = ormi::backends::cpu::jit::cache::has_entries();
            found = static_cast<bool>(ormi::backends::cpu::jit::cache::find("readable", {}));
        });
        std::filesystem::permissions(cache.path, permissions);
        EXPECT_TRUE(possible);
        EXPECT_TRUE(found);
    }
#endif

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

    TEST(CPUJITCacheTest, missingLibraryDoesNotCreateLockFiles) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());

        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("missing", {}));
        EXPECT_FALSE(std::filesystem::exists(cache.path));

        std::filesystem::create_directories(cache.path);
        for (int i = 0; i < 3; ++i) {
            EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("missing", {}));
        }
        EXPECT_TRUE(std::filesystem::is_empty(cache.path));

        std::ofstream(source(cache, "orphan")) << "orphan";
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("orphan", {}));
        auto lock = source(cache, "orphan");
        lock.replace_extension(".lock");
        EXPECT_FALSE(std::filesystem::exists(lock));
        EXPECT_EQ(read(source(cache, "orphan")), "orphan");
    }

    TEST(CPUJITCacheTest, missingLookupFindsLaterPublicationAndDirectoryChanges) {
        Directory cache;
        const auto other = cache.path / "other";
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("later", {}));

        {
            const auto built = ormi::backends::cpu::jit::cache::get("later", build);
            EXPECT_TRUE(ormi::backends::cpu::jit::cache::find("later", {}));
        }
        environment("ORMI_CPU_JIT_CACHE_DIR", other.string().c_str());
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("later", {}));
        EXPECT_FALSE(std::filesystem::exists(other));
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        EXPECT_TRUE(ormi::backends::cpu::jit::cache::find("later", {}));
    }

    TEST(CPUJITCacheTest, findRejectsIncompleteAndMismatchedEntries) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(library(cache, "missing source")) << "library";
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("missing source", {}));

        std::ofstream(source(cache, "wrong source")) << "other source";
        std::ofstream(library(cache, "wrong source")) << "library";
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("wrong source", {}));

        std::filesystem::create_directory(library(cache, "directory"));
        std::ofstream(source(cache, "directory")) << "directory";
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("directory", {}));
    }

    TEST(CPUJITCacheTest, findWaitsForInProgressPublication) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::promise<void> staged;
        std::promise<void> publish;
        auto ready = publish.get_future();
        auto building = std::async(std::launch::async, [&] {
            auto lock = source(cache, "publishing");
            lock.replace_extension(".lock");
            ormi::platform::FileLock lease(lock);
            staged.set_value();
            ready.wait();
            std::ofstream(library(cache, "publishing")) << "library";
            std::ofstream(source(cache, "publishing")) << "publishing";
        });
        staged.get_future().wait();
        auto lookup = std::async(std::launch::async, [] {
            return ormi::backends::cpu::jit::cache::find("publishing", {});
        });
        EXPECT_EQ(lookup.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
        publish.set_value();
        building.get();
        EXPECT_TRUE(lookup.get());
        EXPECT_TRUE(ormi::backends::cpu::jit::cache::find("publishing", {}));
    }

    TEST(CPUJITCacheTest, findsCopiedArtifactWithoutALockFile) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(source(cache, "copied")) << "copied";
        std::ofstream(library(cache, "copied")) << "library";
        auto lock = source(cache, "copied");
        lock.replace_extension(".lock");
        EXPECT_FALSE(std::filesystem::exists(lock));

        const auto found = ormi::backends::cpu::jit::cache::find("copied", {});

        ASSERT_TRUE(found.has_value());
        EXPECT_TRUE(std::filesystem::is_regular_file(lock));
        EXPECT_EQ(found->library, library(cache, "copied"));
    }

    TEST(CPUJITCacheTest, findReportsInvalidCacheDirectory) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::ofstream(cache.path) << "file";

        EXPECT_THROW(ormi::backends::cpu::jit::cache::find("missing", {}), ormi::jitError);
    }

    TEST(CPUJITCacheTest, existingLockDistinguishesMissingFilesFromOpenErrors) {
        Directory cache;
        const auto missing = cache.path / "missing.lock";
        {
            ormi::platform::FileLock lock(
                missing, ormi::platform::Lock::shared, ormi::platform::FileLock::Open::existing
            );
            EXPECT_FALSE(lock.opened());
        }
        EXPECT_FALSE(std::filesystem::exists(cache.path));

        const auto blocked = cache.path / "blocked.lock";
        std::filesystem::create_directories(blocked);
        EXPECT_THROW(
            ormi::platform::FileLock(
                blocked, ormi::platform::Lock::shared, ormi::platform::FileLock::Open::existing
            ),
            ormi::jitError
        );
    }

    TEST(CPUJITCacheTest, findLeaseBlocksDiscard) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        {
            const auto built = ormi::backends::cpu::jit::cache::get("leased find", build);
        }
        auto found = ormi::backends::cpu::jit::cache::find("leased find", {});
        ASSERT_TRUE(found.has_value());
        auto discarding = std::async(std::launch::async, [] {
            ormi::backends::cpu::jit::cache::discard("leased find");
        });

        EXPECT_EQ(discarding.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
        found.reset();
        discarding.get();
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("leased find", {}));
    }

    TEST(CPUJITCacheTest, findValidatesAfterWriterReleasesLock) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        using Artifact = ormi::backends::cpu::jit::cache::Artifact;
        std::future<std::optional<Artifact>> lookup;
        {
            auto lock = source(cache, "writer");
            lock.replace_extension(".lock");
            ormi::platform::FileLock lease(lock);
            std::ofstream(library(cache, "writer")) << "library";
            std::ofstream(source(cache, "writer")) << "incomplete";
            lookup = std::async(std::launch::async, [] {
                return ormi::backends::cpu::jit::cache::find("writer", {});
            });

            EXPECT_EQ(lookup.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
            std::ofstream(source(cache, "writer")) << "writer";
        }
        EXPECT_TRUE(lookup.get());
    }

    TEST(CPUJITCacheTest, publishesStagedSourceAndLibrary) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        builds = 0;
        const auto producer = [](const std::filesystem::path& staged_source,
            const std::filesystem::path& staged_library) {
            ++builds;
            EXPECT_TRUE(staged_source.filename().string().ends_with(".tmp.cpp"));
            EXPECT_EQ(staged_library.stem(), staged_source.stem());
            EXPECT_EQ(staged_library.extension().string(), ormi::platform::shared_extension());
            EXPECT_EQ(read(staged_source), "publication source");
            std::ofstream(staged_library, std::ios::binary) << "published library";
            auto object = staged_source;
            object.replace_extension(".obj");
            std::ofstream(object) << "compiler sidecar";
        };

        const auto artifact = ormi::backends::cpu::jit::cache::get("publication source", producer);

        EXPECT_FALSE(artifact.cached);
        EXPECT_EQ(artifact.library, library(cache, "publication source"));
        EXPECT_EQ(read(source(cache, "publication source")), "publication source");
        EXPECT_EQ(read(artifact.library), "published library");
        expect_no_staging(cache);
        EXPECT_TRUE(ormi::backends::cpu::jit::cache::get("publication source", producer).cached);
        EXPECT_EQ(builds, 1);
    }

    TEST(CPUJITCacheTest, cleansStagingAndReleasesLockWhenBuildThrows) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        builds = 0;
        const auto failing = [](const std::filesystem::path& staged_source,
            const std::filesystem::path& staged_library) {
            ++builds;
            std::ofstream(staged_library, std::ios::binary) << "partial library";
            auto sidecar = staged_source;
            sidecar.replace_extension(".obj");
            std::ofstream(sidecar) << "partial object";
            sidecar.replace_extension(".pdb");
            std::ofstream(sidecar) << "partial debug data";
            throw std::runtime_error("compiler failed");
        };

        EXPECT_THROW(
            ormi::backends::cpu::jit::cache::get("failed build", failing),
            std::runtime_error
        );
        EXPECT_FALSE(std::filesystem::exists(source(cache, "failed build")));
        EXPECT_FALSE(std::filesystem::exists(library(cache, "failed build")));
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("failed build", {}));
        expect_no_staging(cache);

        const auto retried = ormi::backends::cpu::jit::cache::get("failed build", build);
        EXPECT_FALSE(retried.cached);
        EXPECT_EQ(builds, 2);
    }

    TEST(CPUJITCacheTest, rejectsMissingLibraryAndCleansStaging) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        const auto missing = [](const std::filesystem::path& staged_source,
            const std::filesystem::path&) {
            auto object = staged_source;
            object.replace_extension(".obj");
            std::ofstream(object) << "object without library";
        };

        EXPECT_THROW(
            ormi::backends::cpu::jit::cache::get("missing library", missing),
            ormi::jitError
        );
        EXPECT_FALSE(std::filesystem::exists(source(cache, "missing library")));
        EXPECT_FALSE(std::filesystem::exists(library(cache, "missing library")));
        expect_no_staging(cache);
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::get("missing library", build).cached);
    }

    TEST(CPUJITCacheTest, cleansStagingWhenLibraryPublicationFails) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        const auto blocked = library(cache, "blocked library");
        std::filesystem::create_directories(blocked);
        const auto blocker = blocked / "keep";
        std::ofstream(blocker) << "existing file";

        EXPECT_THROW(
            ormi::backends::cpu::jit::cache::get("blocked library", build),
            ormi::jitError
        );
        EXPECT_EQ(read(blocker), "existing file");
        EXPECT_FALSE(std::filesystem::exists(source(cache, "blocked library")));
        expect_no_staging(cache);

        std::filesystem::remove(blocker);
        std::filesystem::remove(blocked);
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::get("blocked library", build).cached);
    }

    TEST(CPUJITCacheTest, doesNotAcceptLibraryWhenSourcePublicationFails) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        const auto blocked = source(cache, "blocked source");
        std::filesystem::create_directories(blocked);
        const auto blocker = blocked / "keep";
        std::ofstream(blocker) << "existing file";

        EXPECT_THROW(
            ormi::backends::cpu::jit::cache::get("blocked source", build),
            ormi::jitError
        );
        EXPECT_TRUE(std::filesystem::is_regular_file(library(cache, "blocked source")));
        EXPECT_EQ(read(blocker), "existing file");
        EXPECT_FALSE(ormi::backends::cpu::jit::cache::find("blocked source", {}));
        expect_no_staging(cache);

        std::filesystem::remove(blocker);
        std::filesystem::remove(blocked);
        const auto retried = ormi::backends::cpu::jit::cache::get("blocked source", build);
        EXPECT_FALSE(retried.cached);
        EXPECT_EQ(read(source(cache, "blocked source")), "blocked source");
        expect_no_staging(cache);
    }

    TEST(CPUJITCacheTest, leavesOtherEntriesStagingFilesUntouched) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        const auto other = cache.path /
            (ormi::backends::cpu::jit::cache::key("other entry") + ".123.tmp.obj");
        std::ofstream(other) << "other compiler's output";

        const auto artifact = ormi::backends::cpu::jit::cache::get("current entry", build);

        EXPECT_FALSE(artifact.cached);
        EXPECT_EQ(read(other), "other compiler's output");
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

    TEST(CPUJITCacheTest, discardRemovesCompletePairAndKeepsLock) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(source(cache, "discard")) << "source";
        std::ofstream(library(cache, "discard")) << "artifact";
        auto lock = source(cache, "discard");
        lock.replace_extension(".lock");

        ormi::backends::cpu::jit::cache::discard("discard");

        EXPECT_FALSE(std::filesystem::exists(library(cache, "discard")));
        EXPECT_FALSE(std::filesystem::exists(source(cache, "discard")));
        EXPECT_TRUE(std::filesystem::is_regular_file(lock));
        EXPECT_EQ(ormi::core::jit::info().bytes, 0);
    }

    TEST(CPUJITCacheTest, discardKeepsSourceWhenLibraryIsMissing) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(source(cache, "orphan")) << "source";

        ormi::backends::cpu::jit::cache::discard("orphan");

        EXPECT_EQ(read(source(cache, "orphan")), "source");
        EXPECT_FALSE(std::filesystem::exists(library(cache, "orphan")));
    }

    TEST(CPUJITCacheTest, discardKeepsSourceWhenLibrarySizeFails) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(library(cache, "blocked"));
        std::ofstream(source(cache, "blocked")) << "source";
        const auto blocker = library(cache, "blocked") / "keep";
        std::ofstream(blocker) << "existing file";

        ormi::backends::cpu::jit::cache::discard("blocked");

        EXPECT_EQ(read(source(cache, "blocked")), "source");
        EXPECT_EQ(read(blocker), "existing file");
    }

    TEST(CPUJITCacheTest, discardAllowsPartialSourceCleanup) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(source(cache, "partial"));
        std::ofstream(library(cache, "partial")) << "artifact";
        const auto blocker = source(cache, "partial") / "keep";
        std::ofstream(blocker) << "existing file";

        ormi::backends::cpu::jit::cache::discard("partial");

        EXPECT_FALSE(std::filesystem::exists(library(cache, "partial")));
        EXPECT_EQ(read(blocker), "existing file");
        EXPECT_EQ(ormi::core::jit::info().bytes, 0);
    }

    TEST(CPUJITCacheTest, clearRemovesOrphanedSourceAndKeepsLock) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(source(cache, "orphan")) << "source";
        auto lock = source(cache, "orphan");
        lock.replace_extension(".lock");

        const auto info = ormi::core::jit::clear();

        EXPECT_EQ(info.entries, 0);
        EXPECT_FALSE(std::filesystem::exists(source(cache, "orphan")));
        EXPECT_TRUE(std::filesystem::is_regular_file(lock));
    }

    TEST(CPUJITCacheTest, clearKeepsSourceWhenLibraryRemovalFailsAndCleansStaging) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(library(cache, "blocked"));
        std::ofstream(source(cache, "blocked")) << "source";
        const auto blocker = library(cache, "blocked") / "keep";
        std::ofstream(blocker) << "existing file";
        const auto staged = cache.path /
            (ormi::backends::cpu::jit::cache::key("blocked") + ".123.tmp.obj");
        std::ofstream(staged) << "staged object";

        const auto info = ormi::core::jit::clear();

        EXPECT_EQ(info.entries, 0);
        EXPECT_EQ(read(source(cache, "blocked")), "source");
        EXPECT_EQ(read(blocker), "existing file");
        EXPECT_FALSE(std::filesystem::exists(staged));
    }

    TEST(CPUJITCacheTest, clearLeavesUnrelatedFilesAndLockFiles) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(source(cache, "complete")) << "source";
        std::ofstream(library(cache, "complete")) << "artifact";
        const auto unrelated = cache.path / "keep.txt";
        std::ofstream(unrelated) << "unrelated file";
        const auto key = ormi::backends::cpu::jit::cache::key("complete");
        const auto lock = cache.path / (key + ".lock");
        const auto staged = cache.path / (key + ".123.tmp.obj");
        std::ofstream(staged) << "staged object";

        const auto info = ormi::core::jit::clear();

        EXPECT_EQ(info.entries, 0);
        EXPECT_EQ(info.bytes, 0);
        EXPECT_EQ(read(unrelated), "unrelated file");
        EXPECT_FALSE(std::filesystem::exists(staged));
        EXPECT_TRUE(std::filesystem::is_regular_file(lock));
        EXPECT_TRUE(std::filesystem::is_regular_file(cache.path / ".maintenance.lock"));
    }

    TEST(CPUJITCacheTest, reportsCompleteEntriesAmongIncompleteFiles) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream(source(cache, "complete")) << "complete";
        std::ofstream(library(cache, "complete")) << "artifact";
        std::ofstream(source(cache, "source only")) << "incomplete";
        std::ofstream(library(cache, "library only")) << "incomplete";
        std::filesystem::create_directory(source(cache, "source directory"));
        std::ofstream(library(cache, "source directory")) << "artifact";
        std::ofstream(source(cache, "library directory")) << "incomplete";
        std::filesystem::create_directory(library(cache, "library directory"));

        const auto info = ormi::core::jit::info();

        EXPECT_EQ(info.entries, 1);
        EXPECT_EQ(info.bytes, 16);
    }

    TEST(CPUJITCacheTest, ignoresUnrelatedAndStagedFiles) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        const auto extension = std::string(ormi::platform::shared_extension());
        for (const auto& stem : {std::string(32, 'z'), std::string(31, 'a'),
            std::string("unrelated")}) {
            std::ofstream(cache.path / (stem + ".cpp")) << "source";
            std::ofstream(cache.path / (stem + extension)) << "library";
        }
        const auto key = ormi::backends::cpu::jit::cache::key("staged");
        std::ofstream(cache.path / (key + ".123.tmp.cpp")) << "staged source";
        std::ofstream(cache.path / (key + ".123.tmp" + extension)) << "staged library";
        std::ofstream(cache.path / (key + ".lock")) << "lock";
        std::ofstream(cache.path / (key + ".obj")) << "object";
        const auto nested = cache.path / "compilers";
        std::filesystem::create_directory(nested);
        std::ofstream(nested / (key + ".cpp")) << "nested source";
        std::ofstream(nested / (key + extension)) << "nested library";

        const auto info = ormi::core::jit::info();

        EXPECT_EQ(info.entries, 0);
        EXPECT_EQ(info.bytes, 0);
    }

    TEST(CPUJITCacheTest, reportsZeroByteCompleteEntries) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        std::ofstream{source(cache, "empty")};
        std::ofstream{library(cache, "empty")};

        const auto info = ormi::core::jit::info();

        EXPECT_EQ(info.entries, 1);
        EXPECT_EQ(info.bytes, 0);
    }

    TEST(CPUJITCacheTest, reportsUppercaseHexKeys) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        std::filesystem::create_directories(cache.path);
        const std::string key(32, 'A');
        std::ofstream(cache.path / (key + ".cpp")) << "source";
        std::ofstream(cache.path / (key + std::string(ormi::platform::shared_extension())))
            << "library";

        const auto info = ormi::core::jit::info();

        EXPECT_EQ(info.entries, 1);
        EXPECT_EQ(info.bytes, 13);
    }

    TEST(CPUJITCacheTest, pruningIgnoresIncompleteAndUnrelatedFiles) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        environment("ORMI_CPU_JIT_CACHE_MAX_BYTES", "0");
        const auto producer = [](const std::filesystem::path&,
            const std::filesystem::path& output) {
            std::ofstream(output) << "artifact";
        };
        ormi::backends::cpu::jit::cache::get("old", producer);
        std::filesystem::last_write_time(source(cache, "old"),
            std::filesystem::file_time_type::clock::now() - std::chrono::hours(2));
        const std::string contents(4096, 'x');
        const auto incomplete = source(cache, "incomplete");
        const auto unrelated = cache.path / "unrelated.cpp";
        std::ofstream(incomplete) << contents;
        std::ofstream(unrelated) << contents;
        environment("ORMI_CPU_JIT_CACHE_MAX_BYTES", "11");

        ormi::backends::cpu::jit::cache::get("new", producer);
        const auto info = ormi::core::jit::info();

        EXPECT_EQ(info.entries, 1);
        EXPECT_EQ(info.bytes, 11);
        EXPECT_FALSE(std::filesystem::exists(source(cache, "old")));
        EXPECT_FALSE(std::filesystem::exists(library(cache, "old")));
        EXPECT_TRUE(std::filesystem::is_regular_file(source(cache, "new")));
        EXPECT_TRUE(std::filesystem::is_regular_file(library(cache, "new")));
        EXPECT_EQ(read(incomplete), contents);
        EXPECT_EQ(read(unrelated), contents);
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

    TEST(CPUJITCacheTest, clearsCompilerProbeRecords) {
        Directory cache;
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.path.string().c_str());
        const auto compilers = cache.path / "compilers";
        std::filesystem::create_directories(compilers);
        const auto accepted = compilers / "accepted.ok";
        const auto rejected = compilers / "rejected.fail";
        std::ofstream(accepted) << "accepted";
        std::ofstream(rejected) << "rejected";

        const auto info = ormi::core::jit::clear();

        EXPECT_EQ(info.entries, 0);
        EXPECT_FALSE(std::filesystem::exists(accepted));
        EXPECT_FALSE(std::filesystem::exists(rejected));
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
        const auto reloaded = ormi::platform::run(helper, load);
        EXPECT_EQ(reloaded.code, 0) << reloaded.output;
    }
}
