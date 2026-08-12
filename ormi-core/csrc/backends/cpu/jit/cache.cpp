#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ormi/core/error.h>

#include "backends/cpu/jit/cache.h"
#include "platform/platform.h"

namespace ormi::backends::cpu::jit::cache {
    namespace {
        constexpr std::string_view abi = "ormi-cpu-jit-v1";
        constexpr std::uintmax_t default_limit = 256ull * 1024ull * 1024ull;

        struct Files {
            std::filesystem::path source;
            std::filesystem::path library;
            std::filesystem::path lock;
        };

        struct Entry {
            std::string key;
            Files files;
            std::uintmax_t bytes;
            std::filesystem::file_time_type used;
        };

        std::string_view architecture() {
#if defined(_M_X64) || defined(__x86_64__)
            return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
            return "aarch64";
#else
            return "unknown";
#endif
        }

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
            result << std::hex << std::setfill('0') << std::setw(16) << value;
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
                root / (std::string(stem) + ".cpp"),
                root / (std::string(stem) + std::string(platform::shared_extension())),
                root / (std::string(stem) + ".lock")
            };
        }

        Files files(std::string_view generated, std::string_view context) {
            return files_for(key(generated, context));
        }

        std::uintmax_t configured_limit() {
            const char* configured = std::getenv("ORMI_CPU_JIT_CACHE_MAX_BYTES");
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
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
                ORMI_THROW(
                    jitError,
                    "ORMI_CPU_JIT_CACHE_MAX_BYTES must be an unsigned decimal byte count."
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
                    "Cannot create CPU JIT cache directory: " + directory().string() +
                        "\n" + error.message()
                );
            }
        }

        bool directory_exists() {
            std::error_code error;
            const auto status = std::filesystem::symlink_status(directory(), error);
            if (error == std::errc::no_such_file_or_directory) {
                return false;
            }
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot inspect CPU JIT cache directory: " + error.message()
                );
            }
            if (!std::filesystem::exists(status)) {
                return false;
            }
            if (!std::filesystem::is_directory(status)) {
                ORMI_THROW(jitError, "CPU JIT cache path is not a directory.");
            }
            return true;
        }

        bool valid(const Files& paths, std::string_view generated) {
            std::error_code error;
            if (
                !std::filesystem::is_regular_file(paths.library, error) ||
                error ||
                !std::filesystem::is_regular_file(paths.source, error) ||
                error
            ) {
                return false;
            }

            const auto size = std::filesystem::file_size(paths.source, error);
            if (error || size != generated.size()) {
                return false;
            }

            std::ifstream source(paths.source, std::ios::binary);
            std::string stored(generated.size(), '\0');
            source.read(stored.data(), static_cast<std::streamsize>(stored.size()));
            return source && stored == generated;
        }

        void touch(const Files& paths) {
            std::error_code ignored;
            std::filesystem::last_write_time(
                paths.source,
                std::filesystem::file_time_type::clock::now(),
                ignored
            );
        }

        void write(
            const std::filesystem::path& path,
            std::string_view generated
        ) {
            std::ofstream source(path, std::ios::binary | std::ios::trunc);
            if (!source) {
                ORMI_THROW(
                    jitError,
                    "Cannot write generated CPU kernel source: " + path.string()
                );
            }
            source.write(
                generated.data(),
                static_cast<std::streamsize>(generated.size())
            );
            if (!source) {
                ORMI_THROW(
                    jitError,
                    "Cannot finish generated CPU kernel source: " + path.string()
                );
            }
        }

        void replace(
            const std::filesystem::path& source,
            const std::filesystem::path& destination
        ) {
            std::error_code error;
            std::filesystem::remove(destination, error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot replace CPU JIT cache entry: " + destination.string()
                );
            }

            std::filesystem::rename(source, destination, error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot publish CPU JIT cache entry: " + destination.string()
                );
            }
        }

        void cleanup(const std::filesystem::path& path) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }

        void cleanup_staged(const std::filesystem::path& source) {
            const auto prefix = source.stem().string() + ".";
            std::error_code error;
            std::filesystem::directory_iterator iterator(source.parent_path(), error);
            const std::filesystem::directory_iterator end;

            while (!error && iterator != end) {
                const auto path = iterator->path();
                if (path.filename().string().starts_with(prefix)) {
                    cleanup(path);
                }
                iterator.increment(error);
            }
        }

        std::vector<Entry> scan(std::error_code& error) {
            std::vector<Entry> result;
            std::filesystem::directory_iterator iterator(directory(), error);
            const std::filesystem::directory_iterator end;

            while (!error && iterator != end) {
                const auto path = iterator->path();
                if (path.extension() == ".cpp") {
                    const auto stem = path.stem().string();
                    if (is_key(stem)) {
                        const auto paths = files_for(stem);
                        std::error_code entry_error;
                        if (
                            std::filesystem::is_regular_file(paths.source, entry_error) &&
                            !entry_error &&
                            std::filesystem::is_regular_file(paths.library, entry_error) &&
                            !entry_error
                        ) {
                            const auto source_bytes = std::filesystem::file_size(
                                paths.source,
                                entry_error
                            );
                            const auto library_bytes = std::filesystem::file_size(
                                paths.library,
                                entry_error
                            );
                            const auto used = std::filesystem::last_write_time(
                                paths.source,
                                entry_error
                            );
                            if (!entry_error) {
                                result.push_back(Entry{
                                    stem,
                                    paths,
                                    source_bytes + library_bytes,
                                    used
                                });
                            }
                        }
                    }
                }
                iterator.increment(error);
            }
            return result;
        }

        Info snapshot(std::uintmax_t limit) {
            std::error_code error;
            const auto exists = std::filesystem::is_directory(directory(), error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot inspect CPU JIT cache directory: " + error.message()
                );
            }
            if (!exists) {
                return Info{directory(), 0, 0, limit};
            }

            const auto entries = scan(error);
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot inspect CPU JIT cache: " + error.message()
                );
            }

            std::uintmax_t bytes = 0;
            for (const auto& entry : entries) {
                bytes += entry.bytes;
            }
            return Info{directory(), entries.size(), bytes, limit};
        }

        std::uintmax_t erase(const Files& paths) {
            std::error_code error;
            const auto library_bytes = std::filesystem::file_size(paths.library, error);
            if (error || !std::filesystem::remove(paths.library, error) || error) {
                return 0;
            }

            auto removed = library_bytes;
            const auto source_bytes = std::filesystem::file_size(paths.source, error);
            if (!error && std::filesystem::remove(paths.source, error) && !error) {
                removed += source_bytes;
            }
            return removed;
        }

        void prune(std::string_view protected_key, std::uintmax_t limit) {
            if (limit == 0) {
                return;
            }

            try {
                platform::FileLock maintenance(directory() / ".maintenance.lock");
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
                    const auto used = std::filesystem::last_write_time(
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
                // Cache maintenance must not invalidate a successfully built kernel.
            }
        }

        std::set<std::string> cache_keys(std::error_code& error) {
            std::set<std::string> result;
            std::filesystem::directory_iterator iterator(directory(), error);
            const std::filesystem::directory_iterator end;

            while (!error && iterator != end) {
                const auto name = iterator->path().filename().string();
                if (
                    name.size() > 32 &&
                    name[32] == '.' &&
                    is_key(std::string_view(name).substr(0, 32))
                ) {
                    result.insert(name.substr(0, 32));
                }
                iterator.increment(error);
            }
            return result;
        }

        void clear_key(std::string_view stem) {
            const auto paths = files_for(stem);
            platform::FileLock lock(paths.lock);

            std::error_code error;
            const bool library_removed =
                !std::filesystem::exists(paths.library, error) ||
                (std::filesystem::remove(paths.library, error) && !error);
            if (library_removed) {
                cleanup(paths.source);
            }

            error.clear();
            std::filesystem::directory_iterator iterator(directory(), error);
            const std::filesystem::directory_iterator end;
            const auto prefix = std::string(stem) + ".";
            while (!error && iterator != end) {
                const auto path = iterator->path();
                const auto name = path.filename().string();
                if (
                    name.starts_with(prefix) &&
                    path != paths.source &&
                    path != paths.library &&
                    path != paths.lock
                ) {
                    cleanup(path);
                }
                iterator.increment(error);
            }
            if (error) {
                ORMI_THROW(
                    jitError,
                    "Cannot clear CPU JIT cache entry: " + error.message()
                );
            }
        }
    }

    Artifact::Artifact(
        std::filesystem::path library_path,
        bool is_cached,
        std::unique_ptr<platform::FileLock> lease
    )
        : library(std::move(library_path)),
          cached(is_cached),
          lease_(std::move(lease)) {}

    Artifact::~Artifact() = default;
    Artifact::Artifact(Artifact&&) noexcept = default;
    Artifact& Artifact::operator=(Artifact&&) noexcept = default;

    std::filesystem::path directory() {
        if (const char* configured = std::getenv("ORMI_CPU_JIT_CACHE_DIR")) {
            if (*configured) {
                return std::filesystem::path(configured);
            }
        }
        return platform::cache_directory() / "jit" / std::string(abi);
    }

    std::string key(std::string_view generated) {
        return key(generated, {});
    }

    std::string key(std::string_view generated, std::string_view context) {
        const auto contract =
            std::string(abi) + "\n" +
            std::string(architecture()) + "\n" +
            std::string(platform::shared_extension()) + "\n" +
            std::string(context) + "\n" +
            std::string(generated);
        return
            hex(hash(contract, 14695981039346656037ull)) +
            hex(hash(contract, 1099511628211ull));
    }

    Artifact get(std::string_view generated, Build build) {
        return get(generated, {}, build);
    }

    Artifact get(
        std::string_view generated,
        std::string_view context,
        Build build
    ) {
        create_directory();
        const auto stem = key(generated, context);
        const auto paths = files_for(stem);
        bool built = false;

        while (true) {
            auto lease = std::make_unique<platform::FileLock>(
                paths.lock,
                platform::Lock::shared
            );
            if (valid(paths, generated)) {
                touch(paths);
                return Artifact{paths.library, !built, std::move(lease)};
            }
            lease.reset();

            std::uintmax_t limit = 0;
            {
                platform::FileLock lock(paths.lock);
                if (valid(paths, generated)) {
                    continue;
                }

                limit = configured_limit();
                const auto suffix =
                    "." + std::to_string(platform::process_id()) + ".tmp";
                const auto staged_stem = paths.source.stem().string() + suffix;
                const auto staged_source = paths.source.parent_path() /
                    (staged_stem + ".cpp");
                const auto staged_library = paths.library.parent_path() /
                    (staged_stem + std::string(platform::shared_extension()));
                cleanup_staged(staged_source);

                try {
                    write(staged_source, generated);
                    build(staged_source, staged_library);
                    if (!std::filesystem::is_regular_file(staged_library)) {
                        ORMI_THROW(
                            jitError,
                            "CPU JIT compiler did not produce a shared library."
                        );
                    }
                    replace(staged_library, paths.library);
                    replace(staged_source, paths.source);
                    cleanup_staged(staged_source);
                } catch (...) {
                    cleanup_staged(staged_source);
                    throw;
                }
            }

            built = true;
            prune(stem, limit);
        }
    }

    std::optional<Artifact> find(
        std::string_view generated,
        std::string_view context
    ) {
        if (!directory_exists()) {
            return std::nullopt;
        }

        const auto paths = files(generated, context);
        auto lease = std::make_unique<platform::FileLock>(
            paths.lock,
            platform::Lock::shared
        );
        if (!valid(paths, generated)) {
            return std::nullopt;
        }

        touch(paths);
        return Artifact{paths.library, true, std::move(lease)};
    }

    void discard(std::string_view generated) {
        discard(generated, {});
    }

    void discard(std::string_view generated, std::string_view context) {
        create_directory();
        const auto paths = files(generated, context);
        platform::FileLock lock(paths.lock);
        erase(paths);
    }

    Info info() {
        const auto limit = configured_limit();
        if (!directory_exists()) {
            return Info{directory(), 0, 0, limit};
        }

        platform::FileLock maintenance(directory() / ".maintenance.lock");
        return snapshot(limit);
    }

    Info clear() {
        const auto limit = configured_limit();
        if (!directory_exists()) {
            return Info{directory(), 0, 0, limit};
        }

        platform::FileLock maintenance(directory() / ".maintenance.lock");
        std::error_code error;
        const auto keys = cache_keys(error);
        if (error) {
            ORMI_THROW(
                jitError,
                "Cannot inspect CPU JIT cache for clearing: " + error.message()
            );
        }
        for (const auto& stem : keys) {
            clear_key(stem);
        }
        return snapshot(limit);
    }
}
