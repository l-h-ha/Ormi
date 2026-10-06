#include <atomic>
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
#include "platform/platform.h"

namespace ormi::backends::cpu::jit::compatibility {
    namespace {
        constexpr std::size_t max_output = 4096;
        constexpr std::string_view source_text =
            "#include <bit>\n"
            "#include <cmath>\n"
            "#include <cstddef>\n"
            "#include <cstdint>\n"
            "extern \"C\" int ormi_compiler_probe() {\n"
            "    return static_cast<int>(sizeof(std::size_t));\n"
            "}\n";
        std::atomic_ullong source_index{0};

        std::filesystem::path probe_directory() {
            return cache::directory() / "compilers";
        }

        class ProbeSource {
        public:
            ProbeSource() {
                std::error_code error;
                const auto temporary = std::filesystem::temp_directory_path(error);
                if (error) {
                    ORMI_THROW(
                        jitError,
                        "Cannot locate the temporary directory for the CPU JIT compiler probe: " +
                            error.message()
                    );
                }

                directory_ = temporary / (
                    "ormi-jit-probe-" +
                    std::to_string(platform::process_id()) + "-" +
                    std::to_string(source_index.fetch_add(1, std::memory_order_relaxed))
                );
                std::filesystem::create_directories(directory_, error);
                if (error) {
                    ORMI_THROW(
                        jitError,
                        "Cannot create the CPU JIT compiler probe directory: " +
                            error.message()
                    );
                }

                std::ofstream file(path(), std::ios::binary | std::ios::trunc);
                file.write(
                    source_text.data(),
                    static_cast<std::streamsize>(source_text.size())
                );
                if (!file) {
                    std::filesystem::remove_all(directory_, error);
                    ORMI_THROW(jitError, "Cannot write the CPU JIT compiler probe source.");
                }
            }

            ~ProbeSource() {
                std::error_code ignored;
                std::filesystem::remove_all(directory_, ignored);
            }

            ProbeSource(const ProbeSource&) = delete;
            ProbeSource& operator=(const ProbeSource&) = delete;

            std::filesystem::path path() const {
                return directory_ / "probe.cpp";
            }

        private:
            std::filesystem::path directory_;
        };

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

        std::vector<std::string> probe_arguments(
            const Toolchain& compiler,
            const std::filesystem::path& source
        ) {
            if (compiler.id == CXX::msvc || compiler.id == CXX::clang_cl) {
                return {"/nologo", "/std:c++20", "/Zs", source.string()};
            }
            return {"-std=c++20", "-fsyntax-only", source.string()};
        }

        std::string failure_message(platform::Process result) {
            std::string reason;
            if (result.state == platform::Process::State::timed_out) {
                reason = "probe timed out";
            } else if (result.state == platform::Process::State::crashed) {
                reason = "probe terminated abnormally with status " +
                    std::to_string(result.code);
            } else {
                reason = "probe failed with exit code " +
                    std::to_string(result.code);
            }
            if (!result.output.empty()) {
                if (result.output.size() > max_output) {
                    result.output.resize(max_output);
                    result.output += "\n[compiler output truncated]";
                }
                reason += "\n" + result.output;
            }
            return reason;
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

        ProbeSource source;
        platform::Process result;
        try {
            result = platform::run(
                compiler.executable,
                probe_arguments(compiler, source.path()),
                compiler.environment,
                timeout
            );
        } catch (const jitError& error) {
            const std::string reason(error.what());
            store_marker(failure_marker, fingerprint + "\n" + reason);
            return reason;
        }

        if (
            result.state == platform::Process::State::exited &&
            result.code == 0
        ) {
            store_marker(success_marker, fingerprint);
            std::filesystem::remove(failure_marker, error);
            return std::nullopt;
        }

        auto reason = failure_message(std::move(result));
        store_marker(failure_marker, fingerprint + "\n" + reason);
        std::filesystem::remove(success_marker, error);
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
