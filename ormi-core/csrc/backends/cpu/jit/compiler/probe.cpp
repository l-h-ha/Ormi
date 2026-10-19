#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/compiler/probe.h"
#include "platform/platform.h"

namespace ormi::backends::cpu::jit::compatibility {
    namespace {
        constexpr std::size_t max_output = 4096;
        std::atomic_ullong source_index{0};

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

    std::optional<std::string> probe(const Toolchain& compiler, std::chrono::milliseconds timeout) {
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
            return std::string(error.what());
        }

        if (result.state == platform::Process::State::exited && result.code == 0) {
            return std::nullopt;
        }
        return failure_message(std::move(result));
    }
}
