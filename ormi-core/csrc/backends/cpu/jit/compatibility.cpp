#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/cache.h"
#include "backends/cpu/jit/compatibility.h"
#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/compiler/probe.h"
#include "platform/platform.h"

namespace ormi::backends::cpu::jit::compatibility {
    namespace {

        std::filesystem::path probe_directory() {
            return cache::directory() / "compilers";
        }

        std::string fingerprint_input(const Toolchain& compiler) {
            std::string result = compiler.executable.string() + "\n" +
                std::string(to_string(compiler.id));

            std::error_code error;
            const auto size = std::filesystem::file_size(
                compiler.executable,
                error
            );
            result += "\n" + (error ? std::string{"?"} : std::to_string(size));
            error.clear();
            const auto modified = std::filesystem::last_write_time(
                compiler.executable,
                error
            );
            result += "\n" + (
                error
                    ? std::string{"?"}
                    : std::to_string(modified.time_since_epoch().count())
            );

            if (!compiler.environment.empty()) {
                for (const auto& variable : compiler.environment) {
                    result += "\n" + variable;
                }
                return result;
            }

            static constexpr const char* inherited[]{
                "PATH",
                "INCLUDE",
                "LIB",
                "LIBPATH",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "COMPILER_PATH",
                "GCC_EXEC_PREFIX",
                "SDKROOT",
                "MACOSX_DEPLOYMENT_TARGET"
            };
            for (const auto* name : inherited) {
                result += "\n";
                result += name;
                result += "=";
                if (const char* value = std::getenv(name)) {
                    result += value;
                }
            }
            return result;
        }

        bool cached_success(
            const std::filesystem::path& marker,
            std::string_view expected
        ) {
            std::error_code error;
            const auto size = std::filesystem::file_size(marker, error);
            if (error || size != expected.size()) {
                return false;
            }

            std::ifstream file(marker, std::ios::binary);
            std::string stored(expected.size(), '\0');
            file.read(stored.data(), static_cast<std::streamsize>(stored.size()));
            return file && stored == expected;
        }

        std::optional<std::string> cached_failure(
            const std::filesystem::path& marker,
            std::string_view expected
        ) {
            std::error_code error;
            const auto size = std::filesystem::file_size(marker, error);
            if (error || size <= expected.size()) {
                return std::nullopt;
            }

            std::ifstream file(marker, std::ios::binary);
            std::string stored(static_cast<std::size_t>(size), '\0');
            file.read(stored.data(), static_cast<std::streamsize>(stored.size()));
            const auto separator = expected.size();
            if (
                !file ||
                stored.compare(0, separator, expected) != 0 ||
                stored[separator] != '\n'
            ) {
                return std::nullopt;
            }
            return stored.substr(separator + 1);
        }

        void store_marker(
            const std::filesystem::path& marker,
            std::string_view value
        ) {
            std::ofstream file(marker, std::ios::binary | std::ios::trunc);
            file.write(value.data(), static_cast<std::streamsize>(value.size()));
            if (!file) {
                ORMI_THROW(
                    jitError,
                    "Cannot cache the CPU JIT compiler probe result: " +
                        marker.string()
                );
            }
        }

    }

    std::optional<std::string> check(
        const Toolchain& compiler,
        std::chrono::milliseconds timeout
    ) {
        const auto fingerprint = cache::key(
            source_text,
            fingerprint_input(compiler)
        );
        const auto probes = probe_directory();
        std::error_code error;
        std::filesystem::create_directories(probes, error);
        if (error) {
            ORMI_THROW(
                jitError,
                "Cannot create the CPU JIT compiler cache: " + error.message()
            );
        }

        platform::FileLock maintenance(
            probes / ".maintenance.lock",
            platform::Lock::shared
        );
        const auto success_marker = probes / (fingerprint + ".ok");
        const auto failure_marker = probes / (fingerprint + ".fail");
        if (cached_success(success_marker, fingerprint)) {
            return std::nullopt;
        }
        if (auto reason = cached_failure(failure_marker, fingerprint)) {
            return reason;
        }

        platform::FileLock lock(probes / (fingerprint + ".lock"));
        if (cached_success(success_marker, fingerprint)) {
            return std::nullopt;
        }
        if (auto reason = cached_failure(failure_marker, fingerprint)) {
            return reason;
        }

        auto reason = probe(compiler, timeout);
        if (reason) {
            store_marker(failure_marker, fingerprint + "\n" + *reason);
            std::filesystem::remove(success_marker, error);
        } else {
            store_marker(success_marker, fingerprint);
            std::filesystem::remove(failure_marker, error);
        }
        return reason;
    }

    void clear() {
        const auto probes = probe_directory();
        std::error_code error;
        const auto status = std::filesystem::symlink_status(probes, error);
        if (error == std::errc::no_such_file_or_directory) {
            return;
        }
        if (error) {
            ORMI_THROW(
                jitError,
                "Cannot inspect the CPU JIT compiler cache: " + error.message()
            );
        }
        if (!std::filesystem::exists(status)) {
            return;
        }
        if (!std::filesystem::is_directory(status)) {
            ORMI_THROW(jitError, "CPU JIT compiler cache path is not a directory.");
        }

        platform::FileLock maintenance(probes / ".maintenance.lock");
        std::vector<std::filesystem::path> markers;
        std::filesystem::directory_iterator iterator(probes, error);
        const std::filesystem::directory_iterator end;
        while (!error && iterator != end) {
            const auto extension = iterator->path().extension();
            if (extension == ".ok" || extension == ".fail") {
                markers.push_back(iterator->path());
            }
            iterator.increment(error);
        }
        if (error) {
            ORMI_THROW(
                jitError,
                "Cannot inspect the CPU JIT compiler cache: " + error.message()
            );
        }

        for (const auto& marker : markers) {
            std::filesystem::remove(marker, error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot clear the CPU JIT compiler cache: " + error.message()
                );
            }
        }
    }
}
