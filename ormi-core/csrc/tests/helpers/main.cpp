#include <cstdlib>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "platform/platform.h"

namespace {
    void environment(const char* name, const std::string& value) {
#if defined(_WIN32)
        _putenv_s(name, value.c_str());
#else
        setenv(name, value.c_str(), 1);
#endif
    }

    void isolate(const char* name) {
        const char* root = std::getenv(name);
        if (!root || *root == '\0') {
            return;
        }

        const auto path = std::filesystem::path(root) /
            std::to_string(ormi::platform::process_id());
        environment(name, path.string());
    }
}

int main(int argc, char** argv) {
    isolate("ORMI_CPU_JIT_CACHE_DIR");
    isolate("ORMI_CUDA_JIT_CACHE_DIR");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
