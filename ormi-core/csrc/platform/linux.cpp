#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <unistd.h>

#include <ormi/core/error.h>

#include "platform/platform.h"

namespace ormi::platform {
    namespace {
        bool executable(const std::filesystem::path& path) {
            return access(path.c_str(), X_OK) == 0;
        }
    }

    std::optional<std::filesystem::path> find(std::string_view name) {
        const std::filesystem::path requested(name);

        if (requested.has_parent_path() || requested.is_absolute()) {
            return executable(requested)
                ? std::optional<std::filesystem::path>(std::filesystem::absolute(requested))
                : std::nullopt;
        }

        const char* path = std::getenv("PATH");
        if (!path) {
            return std::nullopt;
        }

        std::string_view remaining(path);
        while (true) {
            const auto separator = remaining.find(':');
            const auto directory = remaining.substr(0, separator);
            const auto candidate =
                std::filesystem::path(directory.empty() ? "." : directory) /
                requested;

            if (executable(candidate)) {
                return std::filesystem::absolute(candidate);
            }

            if (separator == std::string_view::npos) {
                break;
            }
            remaining.remove_prefix(separator + 1);
        }

        return std::nullopt;
    }

    FileLock::FileLock(const std::filesystem::path& path, Lock mode)
        : FileLock(path, mode, Open::create) {}

    FileLock::FileLock(const std::filesystem::path& path, Lock mode, Open policy) {
        const auto flags = policy == Open::create ? O_CREAT | O_RDWR : O_RDWR;
        const auto handle = open(path.c_str(), flags, 0600);
        if (handle < 0) {
            if (policy == Open::existing && errno == ENOENT) {
                return;
            }
            ORMI_THROW(jitError, "Cannot open cache lock: " + path.string());
        }

        const auto operation = mode == Lock::exclusive ? LOCK_EX : LOCK_SH;
        while (flock(handle, operation) != 0) {
            if (errno != EINTR) {
                close(handle);
                ORMI_THROW(
                    jitError,
                    "Cannot lock cache entry: " + path.string()
                );
            }
        }
        handle_ = handle;
    }

    FileLock::~FileLock() {
        if (handle_ == -1) {
            return;
        }

        const auto handle = static_cast<int>(handle_);
        flock(handle, LOCK_UN);
        close(handle);
    }

    std::optional<Tool> configured_msvc() {
        return std::nullopt;
    }

    std::span<const std::string_view> cxx_candidates() {
        static constexpr std::array names{
            std::string_view{"g++"},
            std::string_view{"clang++"},
            std::string_view{"c++"}
        };
        return names;
    }

    std::vector<std::string> shared_flags() {
        return {"-shared", "-fPIC"};
    }

    std::filesystem::path cache_directory() {
        if (const char* configured = std::getenv("XDG_CACHE_HOME")) {
            if (*configured) {
                return std::filesystem::path(configured) / "ormi";
            }
        }
        if (const char* home = std::getenv("HOME")) {
            if (*home) {
                return std::filesystem::path(home) / ".cache" / "ormi";
            }
        }
        return std::filesystem::temp_directory_path() / "ormi-cache";
    }

    bool supports_avx2() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
#else
        return false;
#endif
    }

    bool supports_fma() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") &&
            __builtin_cpu_supports("fma");
#else
        return false;
#endif
    }

    bool supports_f16c() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") &&
            __builtin_cpu_supports("f16c");
#else
        return false;
#endif
    }

    std::size_t cpu_cores() {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
            return std::max<std::size_t>(
                std::thread::hardware_concurrency(),
                1
            );
        }

        std::set<std::pair<int, int>> cores;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &allowed)) {
                continue;
            }

            const auto topology =
                std::filesystem::path("/sys/devices/system/cpu") /
                ("cpu" + std::to_string(cpu)) /
                "topology";
            std::ifstream package_file(topology / "physical_package_id");
            std::ifstream core_file(topology / "core_id");
            int package = 0;
            int core = 0;
            if (package_file >> package && core_file >> core) {
                cores.emplace(package, core);
            }
        }

        if (!cores.empty()) {
            return cores.size();
        }
        return std::max<std::size_t>(
            std::thread::hardware_concurrency(),
            1
        );
    }

    unsigned long long process_id() {
        return static_cast<unsigned long long>(getpid());
    }
}
