#pragma once
#include <chrono>
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

    class FileLock {
    public:
        explicit FileLock(
            const std::filesystem::path& path,
            Lock mode = Lock::exclusive
        );
        ~FileLock();

        FileLock(const FileLock&) = delete;
        FileLock& operator=(const FileLock&) = delete;

    private:
        std::intptr_t handle_ = -1;
    };

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
    unsigned long long process_id();
}
