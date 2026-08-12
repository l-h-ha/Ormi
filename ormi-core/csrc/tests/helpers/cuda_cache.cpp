#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

#include "backends/cuda/fusion/cache.h"

namespace {
    void environment(const char* name, const char* value) {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        return 2;
    }

    environment("ORMI_CUDA_JIT_CACHE_DIR", argv[1]);
    const bool expected_cached =
        std::string_view(argv[2]) == "cached";

    try {
        const auto artifact =
            ormi::backends::cuda::fusion::cache::get(
                "cross-process-source",
                "cross-process-target",
                [] {
                    return std::string("cross-process-ptx");
                }
            );
        return artifact.cached == expected_cached ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 4;
    }
}
