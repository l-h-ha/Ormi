#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace ormi::platform {
    class FileLock;
}

namespace ormi::backends::cpu::jit::cache {
    class Artifact {
    public:
        Artifact(
            std::filesystem::path library,
            bool cached,
            std::unique_ptr<platform::FileLock> lease
        );
        ~Artifact();

        Artifact(const Artifact&) = delete;
        Artifact& operator=(const Artifact&) = delete;
        Artifact(Artifact&&) noexcept;
        Artifact& operator=(Artifact&&) noexcept;

        std::filesystem::path library;
        bool cached;

    private:
        std::unique_ptr<platform::FileLock> lease_;
    };

    struct Info {
        std::filesystem::path path;
        std::size_t entries;
        std::uintmax_t bytes;
        std::uintmax_t limit;
    };

    using Build = void (*)(
        const std::filesystem::path& source,
        const std::filesystem::path& output
    );

    std::filesystem::path directory();
    std::string key(std::string_view generated);
    Artifact get(std::string_view generated, Build build);
    void discard(std::string_view generated);
    Info info();
    Info clear();
}
