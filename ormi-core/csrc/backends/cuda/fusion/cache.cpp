#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cuda/fusion/cache.h"
#include "platform/platform.h"

namespace ormi::backends::cuda::fusion::cache {
    namespace {
        constexpr std::string_view abi = "ormi-cuda-jit-v1";
        constexpr std::uintmax_t default_limit =
            256ull * 1024ull * 1024ull;

        struct Files {
            std::filesystem::path source;
            std::filesystem::path ptx;
            std::filesystem::path lock;
        };

        struct Entry {
            std::string key;
            Files files;
            std::uintmax_t bytes;
            std::filesystem::file_time_type used;
        };

        std::uint64_t hash(std::string_view value, std::uint64_t seed) {
            auto result = seed;
            for (const auto byte : value) {
                result ^= static_cast<unsigned char>(byte);
                result *= 1099511628211ull;
            }
            return result;
        }

        std::string hex(std::uint64_t value) {
            std::ostringstream result;
            result << std::hex << std::setfill('0') << std::setw(16)
                   << value;
            return result.str();
        }

        bool is_key(std::string_view value) {
            return value.size() == 32 && std::all_of(
                value.begin(),
                value.end(),
                [](unsigned char character) {
                    return std::isxdigit(character) != 0;
                }
            );
        }

        Files files_for(std::string_view stem) {
            const auto root = directory();
            return Files{
                root / (std::string(stem) + ".cu"),
                root / (std::string(stem) + ".ptx"),
                root / (std::string(stem) + ".lock")
            };
        }

        std::uintmax_t configured_limit() {
            const char* configured =
                std::getenv("ORMI_CUDA_JIT_CACHE_MAX_BYTES");
            if (!configured || !*configured) {
                return default_limit;
            }

            std::uintmax_t result = 0;
            const std::string_view value(configured);
            const auto parsed = std::from_chars(
                value.data(),
                value.data() + value.size(),
                result
            );
            if (
                parsed.ec != std::errc{} ||
                parsed.ptr != value.data() + value.size()
            ) {
                ORMI_THROW(
                    jitError,
                    "ORMI_CUDA_JIT_CACHE_MAX_BYTES must be an unsigned "
                    "decimal byte count."
                );
            }
            return result;
        }

        void create_directory() {
            std::error_code error;
            std::filesystem::create_directories(directory(), error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot create CUDA JIT cache directory: " +
                        directory().string() + "\n" + error.message()
                );
            }
        }

        bool directory_exists() {
            std::error_code error;
            const auto status =
                std::filesystem::symlink_status(directory(), error);
            if (error == std::errc::no_such_file_or_directory) {
                return false;
            }
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot inspect CUDA JIT cache directory: " +
                        error.message()
                );
            }
            if (!std::filesystem::exists(status)) {
                return false;
            }
            if (!std::filesystem::is_directory(status)) {
                ORMI_THROW(
                    jitError,
                    "CUDA JIT cache path is not a directory."
                );
            }
            return true;
        }

        void write(
            const std::filesystem::path& path,
            std::string_view value,
            std::string_view description
        ) {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output) {
                ORMI_THROW(
                    jitError,
                    "Cannot write CUDA JIT " + std::string(description) +
                        ": " + path.string()
                );
            }
            output.write(
                value.data(),
                static_cast<std::streamsize>(value.size())
            );
            if (!output) {
                ORMI_THROW(
                    jitError,
                    "Cannot finish CUDA JIT " + std::string(description) +
                        ": " + path.string()
                );
            }
        }

        std::string read(const std::filesystem::path& path) {
            std::ifstream input(path, std::ios::binary);
            if (!input) {
                return {};
            }
            return std::string(
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>()
            );
        }

        bool valid(const Files& paths, std::string_view source) {
            std::error_code error;
            if (
                !std::filesystem::is_regular_file(paths.source, error) ||
                error ||
                !std::filesystem::is_regular_file(paths.ptx, error) ||
                error ||
                std::filesystem::file_size(paths.source, error) !=
                    source.size() ||
                error
            ) {
                return false;
            }
            return read(paths.source) == source &&
                std::filesystem::file_size(paths.ptx, error) != 0 &&
                !error;
        }

        void touch(const Files& paths) {
            std::error_code ignored;
            std::filesystem::last_write_time(
                paths.source,
                std::filesystem::file_time_type::clock::now(),
                ignored
            );
        }

        void cleanup(const std::filesystem::path& path) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }

        void cleanup_staged(const Files& files) {
            const auto prefix = files.source.stem().string() + ".";
            std::error_code error;
            std::filesystem::directory_iterator iterator(
                files.source.parent_path(),
                error
            );
            const std::filesystem::directory_iterator end;
            while (!error && iterator != end) {
                const auto path = iterator->path();
                const auto name = path.filename().string();
                if (
                    name.starts_with(prefix) &&
                    (
                        name.ends_with(".tmp.cu") ||
                        name.ends_with(".tmp.ptx")
                    )
                ) {
                    cleanup(path);
                }
                iterator.increment(error);
            }
        }

        void replace(
            const std::filesystem::path& staged,
            const std::filesystem::path& published
        ) {
            std::error_code error;
            std::filesystem::remove(published, error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot replace CUDA JIT cache entry: " +
                        published.string()
                );
            }
            std::filesystem::rename(staged, published, error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot publish CUDA JIT cache entry: " +
                        published.string()
                );
            }
        }

        std::string publish(const Files& files, std::string_view source, const Build& build) {
            const auto suffix =
                "." + std::to_string(platform::process_id()) + ".tmp";
            const auto staged_source =
                files.source.parent_path() /
                (files.source.stem().string() + suffix + ".cu");
            const auto staged_ptx =
                files.ptx.parent_path() /
                (files.ptx.stem().string() + suffix + ".ptx");
            cleanup_staged(files);

            try {
                auto ptx = build();
                if (ptx.empty()) {
                    ORMI_THROW(
                        jitError,
                        "CUDA JIT compiler produced empty PTX."
                    );
                }
                write(staged_source, source, "source");
                write(staged_ptx, ptx, "PTX");
                replace(staged_ptx, files.ptx);
                replace(staged_source, files.source);
                cleanup_staged(files);
                return ptx;
            } catch (...) {
                cleanup_staged(files);
                throw;
            }
        }

        std::vector<Entry> scan(std::error_code& error) {
            std::vector<Entry> result;
            std::filesystem::directory_iterator iterator(directory(), error);
            const std::filesystem::directory_iterator end;
            for (; !error && iterator != end; iterator.increment(error)) {
                const auto path = iterator->path();
                const auto stem = path.stem().string();
                if (path.extension() != ".cu" || !is_key(stem)) {
                    continue;
                }

                const auto files = files_for(stem);
                std::error_code entry_error;
                const auto source_regular =
                    std::filesystem::is_regular_file(files.source, entry_error);
                if (entry_error || !source_regular) {
                    continue;
                }
                const auto ptx_regular =
                    std::filesystem::is_regular_file(files.ptx, entry_error);
                if (entry_error || !ptx_regular) {
                    continue;
                }

                const auto source_bytes =
                    std::filesystem::file_size(files.source, entry_error);
                if (entry_error) {
                    continue;
                }
                const auto ptx_bytes =
                    std::filesystem::file_size(files.ptx, entry_error);
                if (entry_error) {
                    continue;
                }
                const auto used =
                    std::filesystem::last_write_time(files.source, entry_error);
                if (entry_error) {
                    continue;
                }

                result.push_back(Entry{
                    stem,
                    files,
                    source_bytes + ptx_bytes,
                    used
                });
            }
            return result;
        }

        std::uintmax_t erase(const Files& files) {
            std::uintmax_t removed = 0;
            std::error_code error;
            const auto source_bytes = std::filesystem::file_size(files.source, error);
            if (!error) {
                const auto source_removed = std::filesystem::remove(files.source, error);
                if (!error && source_removed) {
                    removed += source_bytes;
                }
            }

            error.clear();
            const auto ptx_bytes = std::filesystem::file_size(files.ptx, error);
            if (error) {
                return removed;
            }
            const auto ptx_removed = std::filesystem::remove(files.ptx, error);
            if (error || !ptx_removed) {
                return removed;
            }
            return removed + ptx_bytes;
        }

        void prune(std::string_view protected_key, std::uintmax_t limit) {
            if (limit == 0) {
                return;
            }

            try {
                platform::FileLock maintenance(
                    directory() / ".maintenance.lock"
                );
                std::error_code error;
                auto entries = scan(error);
                if (error) {
                    return;
                }

                std::uintmax_t bytes = 0;
                for (const auto& entry : entries) {
                    bytes += entry.bytes;
                }
                std::sort(
                    entries.begin(),
                    entries.end(),
                    [](const Entry& lhs, const Entry& rhs) {
                        return lhs.used < rhs.used;
                    }
                );
                for (const auto& entry : entries) {
                    if (bytes <= limit) {
                        break;
                    }
                    if (entry.key == protected_key) {
                        continue;
                    }

                    platform::FileLock lock(entry.files.lock);
                    const auto used =
                        std::filesystem::last_write_time(
                            entry.files.source,
                            error
                        );
                    if (error || used != entry.used) {
                        error.clear();
                        continue;
                    }

                    const auto removed = erase(entry.files);
                    bytes = removed > bytes ? 0 : bytes - removed;
                }
            } catch (const jitError&) {
                // Maintenance cannot invalidate a successfully built kernel.
            }
        }

        Info snapshot(std::uintmax_t limit) {
            std::error_code error;
            const auto status =
                std::filesystem::symlink_status(directory(), error);
            if (error == std::errc::no_such_file_or_directory) {
                return Info{directory(), 0, 0, limit};
            }
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot inspect CUDA JIT cache directory: " +
                        error.message()
                );
            }
            if (!std::filesystem::exists(status)) {
                return Info{directory(), 0, 0, limit};
            }
            if (!std::filesystem::is_directory(status)) {
                ORMI_THROW(
                    jitError,
                    "CUDA JIT cache path is not a directory."
                );
            }
            const auto entries = scan(error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot inspect CUDA JIT cache: " + error.message()
                );
            }

            std::uintmax_t bytes = 0;
            for (const auto& entry : entries) {
                bytes += entry.bytes;
            }
            return Info{directory(), entries.size(), bytes, limit};
        }
    }

    std::filesystem::path directory() {
        if (const char* configured =
                std::getenv("ORMI_CUDA_JIT_CACHE_DIR")) {
            if (*configured) {
                return std::filesystem::path(configured);
            }
        }
        return platform::cache_directory() / "jit" / std::string(abi);
    }

    std::string key(
        std::string_view source,
        std::string_view context
    ) {
        const auto contract =
            std::string(abi) + "\n" +
            std::string(context) + "\n" +
            std::string(source);
        return
            hex(hash(contract, 14695981039346656037ull)) +
            hex(hash(contract, 1099511628211ull));
    }

    Artifact get(
        std::string_view source,
        std::string_view context,
        const Build& build
    ) {
        create_directory();
        const auto stem = key(source, context);
        const auto files = files_for(stem);
        std::uintmax_t limit = 0;
        std::string ptx;

        {
            platform::FileLock lock(files.lock);
            if (valid(files, source)) {
                ptx = read(files.ptx);
                if (!ptx.empty()) {
                    touch(files);
                    return Artifact{std::move(ptx), true};
                }
            }

            limit = configured_limit();
            ptx = publish(files, source, build);
        }

        prune(stem, limit);
        return Artifact{std::move(ptx), false};
    }

    Info info() {
        const auto limit = configured_limit();
        if (!directory_exists()) {
            return Info{directory(), 0, 0, limit};
        }
        platform::FileLock maintenance(
            directory() / ".maintenance.lock"
        );
        return snapshot(limit);
    }

    Info clear() {
        const auto limit = configured_limit();
        if (!directory_exists()) {
            return Info{directory(), 0, 0, limit};
        }

        platform::FileLock maintenance(
            directory() / ".maintenance.lock"
        );
        std::error_code error;
        const auto entries = scan(error);
        if (error) {
            ORMI_THROW(
                jitError,
                "Cannot inspect CUDA JIT cache for clearing: " +
                    error.message()
            );
        }
        for (const auto& entry : entries) {
            platform::FileLock lock(entry.files.lock);
            cleanup(entry.files.source);
            cleanup(entry.files.ptx);
        }
        return snapshot(limit);
    }
}
