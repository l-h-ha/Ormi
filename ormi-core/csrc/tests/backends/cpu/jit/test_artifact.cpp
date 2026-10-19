#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/jit/artifact.h"
#include "backends/cpu/jit/cache.h"
#include "platform/platform.h"

namespace {
    namespace artifact = ormi::backends::cpu::jit::artifact;
    namespace cache = ormi::backends::cpu::jit::cache;

    // Real OS locks and libraries, with a controlled cache result after lookup misses.
    class CPUJITArtifactTest : public testing::Test {
    public:
        enum class Failure { none, library, symbols };

        inline static CPUJITArtifactTest* current = nullptr;
        std::filesystem::path directory;
        std::filesystem::path library;
        std::filesystem::path lock;
        Failure failure = Failure::none;
        bool lookup_hit = false;
        bool acquired_cached = false;
        bool repair_valid = true;
        int acquisitions = 0;
        int discards = 0;

        cache::Artifact acquire(bool cached, Failure problem) {
            if (problem == Failure::library) {
                std::ofstream(library, std::ios::binary) << "invalid library";
            } else {
                const auto fixture = problem == Failure::symbols
                    ? ORMI_JIT_ARTIFACT_INCOMPLETE
                    : ORMI_JIT_ARTIFACT_FIXTURE;
                std::filesystem::copy_file(fixture, library,
                    std::filesystem::copy_options::overwrite_existing);
            }
            return cache::Artifact(library, cached,
                std::make_unique<ormi::platform::FileLock>(
                    lock, ormi::platform::Lock::shared));
        }

    protected:
        void SetUp() override {
            directory = std::filesystem::temp_directory_path() /
                ("ormi-jit-artifact-test-" +
                    std::to_string(ormi::platform::process_id()));
            std::filesystem::create_directories(directory);
            library = directory / ("kernel" +
                std::string(ormi::platform::shared_extension()));
            lock = directory / "kernel.lock";
            current = this;
        }

        void TearDown() override {
            current = nullptr;
            std::error_code ignored;
            std::filesystem::remove_all(directory, ignored);
        }
    };

    constexpr artifact::Kernel kernel{7, 11, 3, 5, false, false};

    void expect_output(const std::shared_ptr<ormi::backends::cpu::jit::Compiled>& compiled,
        float increment) {
        const std::array<float, 3> input{0.25f, -2.0f, 4.0f};
        std::array<float, 3> output{};
        void* outputs[]{output.data()};
        const void* inputs[]{input.data()};
        compiled->function(outputs, inputs, 0, input.size(), input.size(), compiled->math);
        for (std::size_t i = 0; i < input.size(); ++i) {
            EXPECT_FLOAT_EQ(output[i], input[i] + increment);
        }
    }

    TEST_F(CPUJITArtifactTest, rejectsEmptyRequest) {
        EXPECT_THROW(artifact::get("test source", {}), ormi::OperationError);
        EXPECT_EQ(acquisitions, 0);
        EXPECT_EQ(discards, 0);
    }

    TEST_F(CPUJITArtifactTest, loadsInitialCacheHitWithoutBuilding) {
        lookup_hit = true;
        const auto result = artifact::get("test source", std::span(&kernel, 1));

        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(acquisitions, 0);
        EXPECT_EQ(discards, 0);
        expect_output(result.front(), 1.0f);
    }

    TEST_F(CPUJITArtifactTest, releasesLookupLeaseBeforeDiscard) {
        lookup_hit = true;
        failure = Failure::library;
        const auto result = artifact::get("test source", std::span(&kernel, 1));

        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(acquisitions, 1);
        EXPECT_EQ(discards, 1);
        expect_output(result.front(), 1.0f);
    }

    TEST_F(CPUJITArtifactTest, releasesAcquisitionLeaseBeforeRepair) {
        acquired_cached = true;
        failure = Failure::library;
        const auto result = artifact::get("test source", std::span(&kernel, 1));

        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(acquisitions, 2);
        EXPECT_EQ(discards, 1);
        expect_output(result.front(), 1.0f);
    }

    TEST_F(CPUJITArtifactTest, releasesPartiallyLoadedBatchBeforeRepair) {
        acquired_cached = true;
        failure = Failure::symbols;
        const std::array kernels{kernel, artifact::Kernel{13, 17, 2, 19, true, false}};
        const auto result = artifact::get("test source", kernels);

        ASSERT_EQ(result.size(), 2);
        EXPECT_EQ(acquisitions, 2);
        EXPECT_EQ(discards, 1);
        EXPECT_EQ(result[0]->library, result[1]->library);
        EXPECT_EQ(result[0]->work, 7);
        EXPECT_EQ(result[0]->unit, 3);
        EXPECT_EQ(result[0]->range, 5);
        EXPECT_FALSE(result[0]->partial);
        EXPECT_EQ(result[1]->work, 13);
        EXPECT_EQ(result[1]->unit, 2);
        EXPECT_EQ(result[1]->range, 19);
        EXPECT_TRUE(result[1]->partial);
        expect_output(result[0], 2.0f);
        expect_output(result[1], 3.0f);
    }

    TEST_F(CPUJITArtifactTest, propagatesFreshLibraryFailureWithoutRetry) {
        failure = Failure::library;
        EXPECT_THROW(artifact::get("test source", std::span(&kernel, 1)), ormi::jitError);
        EXPECT_EQ(acquisitions, 1);
        EXPECT_EQ(discards, 0);
        ormi::platform::FileLock released(lock);
    }

    TEST_F(CPUJITArtifactTest, propagatesFailedRepairWithoutAnotherRetry) {
        acquired_cached = true;
        failure = Failure::library;
        repair_valid = false;
        EXPECT_THROW(artifact::get("test source", std::span(&kernel, 1)), ormi::jitError);
        EXPECT_EQ(acquisitions, 2);
        EXPECT_EQ(discards, 1);
        ormi::platform::FileLock released(lock);
    }

    TEST_F(CPUJITArtifactTest, retainsLoadedCodeAfterReleasingCacheLease) {
        lookup_hit = true;
        const auto result = artifact::get("test source", std::span(&kernel, 1));

        ASSERT_EQ(result.size(), 1);
        {
            ormi::platform::FileLock released(lock);
            expect_output(result.front(), 1.0f);
        }
        expect_output(result.front(), 1.0f);
    }
}

// These definitions replace disk acquisition only in the dedicated test executable.
namespace ormi::backends::cpu::jit::cache {
    std::optional<Artifact> find(std::string_view, std::string_view) {
        auto& test = *CPUJITArtifactTest::current;
        if (!test.lookup_hit) {
            return std::nullopt;
        }
        return test.acquire(true, test.failure);
    }

    Artifact get(std::string_view, std::string_view, Build) {
        auto& test = *CPUJITArtifactTest::current;
        ++test.acquisitions;
        auto failure = CPUJITArtifactTest::Failure::none;
        if (test.acquisitions == 1 && test.discards == 0) {
            failure = test.failure;
        } else if (!test.repair_valid) {
            failure = CPUJITArtifactTest::Failure::library;
        }
        return test.acquire(test.acquisitions == 1 && test.acquired_cached, failure);
    }

    void discard(std::string_view, std::string_view) {
        auto& test = *CPUJITArtifactTest::current;
        platform::FileLock exclusive(test.lock);
        std::filesystem::remove(test.library);
        ++test.discards;
        test.lookup_hit = false;
    }
}
