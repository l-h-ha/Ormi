#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "platform/platform.h"
#include "platform/windows/utf.h"

namespace ormi::platform {
    namespace {
        std::string trim(std::string value) {
            while (
                !value.empty() &&
                std::isspace(static_cast<unsigned char>(value.back()))
            ) {
                value.pop_back();
            }
            const auto first = std::find_if_not(
                value.begin(),
                value.end(),
                [](unsigned char character) {
                    return std::isspace(character);
                }
            );
            value.erase(value.begin(), first);
            return value;
        }

        std::filesystem::path environment_path(const wchar_t* name) {
            std::vector<wchar_t> value(32768);
            const auto size = GetEnvironmentVariableW(
                name,
                value.data(),
                static_cast<DWORD>(value.size())
            );
            if (size == 0 || size >= value.size()) {
                return {};
            }
            return std::filesystem::path(value.data(), value.data() + size);
        }

        std::filesystem::path short_path(const std::filesystem::path& path) {
            std::vector<wchar_t> value(32768);
            const auto size = GetShortPathNameW(
                path.c_str(),
                value.data(),
                static_cast<DWORD>(value.size())
            );
            if (size == 0 || size >= value.size()) {
                return {};
            }
            return std::filesystem::path(value.data(), value.data() + size);
        }

        std::vector<std::string> parse_environment(std::string_view output) {
            std::vector<std::string> result;
            std::unordered_set<std::string> keys;
            while (!output.empty()) {
                const auto newline = output.find('\n');
                auto line = output.substr(0, newline);
                if (!line.empty() && line.back() == '\r') {
                    line.remove_suffix(1);
                }
                const auto separator = line.empty()
                    ? std::string_view::npos
                    : line.find('=', line.front() == '=' ? 1 : 0);
                if (separator != std::string_view::npos) {
                    std::string key(line.substr(0, separator));
                    std::transform(
                        key.begin(),
                        key.end(),
                        key.begin(),
                        [](unsigned char character) {
                            return static_cast<char>(std::tolower(character));
                        }
                    );
                    if (keys.insert(std::move(key)).second) {
                        result.emplace_back(line);
                    }
                }

                if (newline == std::string_view::npos) {
                    break;
                }
                output.remove_prefix(newline + 1);
            }
            return result;
        }

        std::optional<std::string_view> environment_value(
            const std::vector<std::string>& environment,
            std::string_view requested
        ) {
            for (const auto& variable : environment) {
                const auto separator = variable.find('=');
                if (separator != requested.size()) {
                    continue;
                }

                bool equal = true;
                for (std::size_t i = 0; i < requested.size(); ++i) {
                    const auto lhs = static_cast<unsigned char>(variable[i]);
                    const auto rhs = static_cast<unsigned char>(requested[i]);
                    if (std::tolower(lhs) != std::tolower(rhs)) {
                        equal = false;
                        break;
                    }
                }
                if (equal) {
                    return std::string_view(variable).substr(separator + 1);
                }
            }
            return std::nullopt;
        }
    }

    std::optional<Tool> configured_msvc() {
        static const auto configured = []() -> std::optional<Tool> {
            const auto program_files = environment_path(L"ProgramFiles(x86)");
            if (program_files.empty()) {
                return std::nullopt;
            }

            const auto vswhere = program_files /
                "Microsoft Visual Studio" /
                "Installer" /
                "vswhere.exe";
            if (!std::filesystem::is_regular_file(vswhere)) {
                return std::nullopt;
            }

            const auto installation = run(
                vswhere,
                std::vector<std::string>{
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath"
                }
            );
            if (installation.code != 0) {
                return std::nullopt;
            }

            const auto root = std::filesystem::path(
                windows::widen(trim(installation.output))
            );
            const auto setup = root / "Common7" / "Tools" / "VsDevCmd.bat";
            const auto shortened = short_path(setup);
            const auto command = find("cmd.exe");
            if (
                root.empty() ||
                shortened.empty() ||
                !std::filesystem::is_regular_file(setup) ||
                !command
            ) {
                return std::nullopt;
            }

            const auto environment_result = run(
                *command,
                std::vector<std::string>{
                    "/d",
                    "/s",
                    "/c",
                    "call " + shortened.string() +
                        " -arch=x64 -host_arch=x64 >nul && set"
                }
            );
            if (environment_result.code != 0) {
                return std::nullopt;
            }

            auto environment = parse_environment(environment_result.output);
            const auto tools = environment_value(
                environment,
                "VCToolsInstallDir"
            );
            if (!tools) {
                return std::nullopt;
            }

            const auto compiler = std::filesystem::path(
                windows::widen(*tools)
            ) / "bin" / "Hostx64" / "x64" / "cl.exe";
            if (!std::filesystem::is_regular_file(compiler)) {
                return std::nullopt;
            }
            return Tool{compiler, std::move(environment)};
        }();

        return configured;
    }

    std::span<const std::string_view> cxx_candidates() {
        static constexpr std::array names{
            std::string_view{"cl.exe"},
            std::string_view{"clang-cl.exe"},
            std::string_view{"g++.exe"},
            std::string_view{"clang++.exe"}
        };
        return names;
    }

    std::vector<std::string> shared_flags() {
        return {"-shared", "-static-libgcc", "-static-libstdc++"};
    }
}
