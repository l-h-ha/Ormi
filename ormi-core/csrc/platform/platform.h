#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ormi::platform {
    enum class Lock {
        shared,
        exclusive
    };

    // Check state first. Read the exit code only for a normal compiler exit.
    struct Process {
        enum class State {
            exited,
            timed_out,
            crashed
        };

        int code;
        std::string output;
        State state;
    };

    struct Tool {
        std::filesystem::path executable;
        std::vector<std::string> environment;
    };

    // Holds an interprocess file lock until destruction. Release a shared lock before
    // requesting an exclusive lock on the same cache entry. Lock upgrades are unsupported.
    class FileLock {
    public:
        enum class Open {
            create,
            existing
        };

        explicit FileLock(
            const std::filesystem::path& path,
            Lock mode = Lock::exclusive
        );
        FileLock(
            const std::filesystem::path& path,
            Lock mode,
            Open policy
        );
        ~FileLock();

        // Opening an absent file with Open::existing leaves the lock unopened.
        bool opened() const noexcept { return handle_ != -1; }

        FileLock(const FileLock&) = delete;
        FileLock& operator=(const FileLock&) = delete;

    private:
        std::intptr_t handle_ = -1;
    };

    // Owns one loaded shared library. Symbol pointers are valid only while it remains loaded.
    class Library {
    public:
        explicit Library(const std::filesystem::path& path);
        ~Library();

        Library(const Library&) = delete;
        Library& operator=(const Library&) = delete;
        Library(Library&& other) noexcept;
        Library& operator=(Library&& other) noexcept;

        void* symbol(std::string_view name) const;

    private:
        void* handle_;
    };

    std::optional<std::filesystem::path> find(std::string_view executable);
    // Execute directly, without a shell. arguments excludes the executable name.
    // An empty environment inherits the parent's environment. A nonempty list replaces it.
    Process run(
        const std::filesystem::path& executable,
        std::span<const std::string> arguments,
        std::span<const std::string> environment = {},
        std::chrono::milliseconds timeout = (std::chrono::milliseconds::max)()
    );
    std::optional<Tool> configured_msvc();
    std::span<const std::string_view> cxx_candidates();
    std::vector<std::string> shared_flags();
    std::string_view shared_extension();
    std::filesystem::path cache_directory();
    std::filesystem::path library_path(const void* anchor, std::string_view stem);
    bool supports_avx2() noexcept;
    bool supports_fma() noexcept;
    bool supports_f16c() noexcept;
    std::size_t cpu_cores();
    unsigned long long process_id();
}
