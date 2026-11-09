#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "platform/platform.h"

namespace {
    const std::filesystem::path fixture(ORMI_LIBRARY_FIXTURE);
    // Static storage stays valid even if a failed ownership test leaks a handle.
    unsigned unloaded = 0;

    int value(const ormi::platform::Library& library) {
        const auto function = reinterpret_cast<int (*)()>(
            library.symbol("ormi_library_value")
        );
        return function();
    }

    void watch_unload(const ormi::platform::Library& library) {
        const auto function = reinterpret_cast<void (*)(unsigned*)>(
            library.symbol("ormi_library_watch_unload")
        );
        unloaded = 0;
        function(&unloaded);
    }

    TEST(PlatformLibraryTest, loadsAndCallsSymbol) {
        const ormi::platform::Library library(fixture);
        EXPECT_EQ(value(library), 42);
    }

    TEST(PlatformLibraryTest, reportsMissingLibrary) {
        const auto missing = fixture / "not-a-library";
        try {
            const ormi::platform::Library library(missing);
            FAIL() << "Loading a missing library should fail.";
        } catch (const ormi::jitError& error) {
            EXPECT_EQ(std::string(error.what()).find(
                "Cannot load shared library: " + missing.string()
            ), 0);
        }
    }

    TEST(PlatformLibraryTest, reportsMissingSymbolAndRecovers) {
        const ormi::platform::Library library(fixture);
        try {
            library.symbol("ormi_library_missing");
            FAIL() << "Looking up a missing symbol should fail.";
        } catch (const ormi::jitError& error) {
            EXPECT_EQ(std::string(error.what()).find(
                "Shared library does not export symbol: ormi_library_missing"
            ), 0);
        }
        EXPECT_EQ(value(library), 42);
    }

    TEST(PlatformLibraryTest, respectsSymbolViewBoundary) {
        const ormi::platform::Library library(fixture);
        constexpr std::string_view text = "ormi_library_value:ignored";
        const auto function = reinterpret_cast<int (*)()>(
            library.symbol(text.substr(0, text.find(':')))
        );
        EXPECT_EQ(function(), 42);
    }

    TEST(PlatformLibraryTest, unloadsOnDestruction) {
        {
            const ormi::platform::Library library(fixture);
            watch_unload(library);
            EXPECT_EQ(unloaded, 0);
        }
        EXPECT_EQ(unloaded, 1);
    }

    TEST(PlatformLibraryTest, moveConstructionTransfersOwnership) {
        std::optional<ormi::platform::Library> destination;
        {
            ormi::platform::Library source(fixture);
            watch_unload(source);
            destination.emplace(std::move(source));
        }
        ASSERT_EQ(unloaded, 0);
        EXPECT_EQ(value(*destination), 42);
        destination.reset();
        EXPECT_EQ(unloaded, 1);
    }

    TEST(PlatformLibraryTest, moveAssignmentReleasesOldOwnership) {
        {
            ormi::platform::Library destination(fixture);
            watch_unload(destination);
            {
                ormi::platform::Library source(fixture);
                destination = std::move(source);
            }
            ASSERT_EQ(unloaded, 0);
            EXPECT_EQ(value(destination), 42);
        }
        EXPECT_EQ(unloaded, 1);
    }

    TEST(PlatformLibraryTest, selfMovePreservesOwnership) {
        {
            ormi::platform::Library library(fixture);
            watch_unload(library);
            auto& same = library;
            library = std::move(same);
            ASSERT_EQ(unloaded, 0);
            EXPECT_EQ(value(library), 42);
        }
        EXPECT_EQ(unloaded, 1);
    }

    TEST(PlatformLibraryTest, locatesModuleFromSymbol) {
        const ormi::platform::Library library(fixture);
        const auto* symbol = library.symbol("ormi_library_value");
        EXPECT_EQ(
            ormi::platform::library_path(symbol, "ormi_library_fixture"), fixture
        );
        EXPECT_EQ(fixture.extension().string(), ormi::platform::shared_extension());
    }
}
