#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/jit.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "platform/platform.h"

namespace {
    void environment(const char* name, const char* value) {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        if (*value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
#endif
    }

    template <typename T>
    bool execute(ormi::core::DType dtype) {
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            dtype
        );
        auto* values = static_cast<T*>(input->data());
        values[0] = static_cast<T>(-0.5);
        values[1] = static_cast<T>(0.0);
        values[2] = static_cast<T>(0.5);
        values[3] = static_cast<T>(1.0);

        const auto square = ormi::core::ops::mul(input, input);
        const auto summed = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::exp(summed)
        );
        const auto* result = static_cast<const T*>(output->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto expected = std::sin(
                std::exp(values[i] * values[i] + values[i])
            );
            const auto tolerance = sizeof(T) == sizeof(float)
                ? static_cast<T>(1.0e-6)
                : static_cast<T>(1.0e-12);
            if (std::abs(result[i] - expected) > tolerance) {
                std::cerr
                    << "incorrect value at index " << i
                    << ": expected " << expected
                    << ", received " << result[i] << '\n';
                return false;
            }
        }
        return true;
    }

    int work(int argc, char** argv, std::string_view mode) {
        const std::filesystem::path cache(argv[2]);
        environment("ORMI_CPU_JIT_CACHE_DIR", argv[2]);

        if (mode == "compile" && argc == 5) {
            environment("ORMI_CPU_JIT_CXX", argv[3]);
            environment("ORMI_CPU_JIT_CXX_ID", argv[4]);
        } else if (mode == "cached" && argc == 3) {
            environment("ORMI_CPU_JIT_CXX", "Z:/ormi/missing/compiler.exe");
            environment("ORMI_CPU_JIT_CXX_ID", "gcc");
        } else {
            std::cerr << "compile requires COMPILER and ID; cached does not\n";
            return 2;
        }

        if (
            !execute<float>(ormi::core::DType::float32) ||
            !execute<double>(ormi::core::DType::float64)
        ) {
            return 4;
        }

        const auto info = ormi::core::jit::info();
        if (info.entries != 2) {
            std::cerr
                << "expected two JIT cache entries, found "
                << info.entries << '\n';
            return 5;
        }

        std::cout << "mode=" << mode << " entries=" << info.entries;
        if (mode == "compile") {
            std::cout << " compiler=" << argv[3] << " id=" << argv[4];
        }
        std::cout << '\n';
        return 0;
    }

    int verify(char** argv) {
        const auto nonce = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()
        );
        const auto cache = std::filesystem::path(argv[2]) /
            (std::string(argv[4]) + "-" + nonce);
        const auto executable = std::filesystem::absolute(argv[0]);
        const auto compile = ormi::platform::run(
            executable,
            std::vector<std::string>{
                "compile",
                cache.string(),
                argv[3],
                argv[4]
            }
        );
        if (
            compile.state != ormi::platform::Process::State::exited ||
            compile.code != 0
        ) {
            std::cerr << "uncached worker failed:\n" << compile.output;
            return 7;
        }

        const auto cached = ormi::platform::run(
            executable,
            std::vector<std::string>{"cached", cache.string()}
        );
        if (
            cached.state != ormi::platform::Process::State::exited ||
            cached.code != 0
        ) {
            std::cerr << "cached worker failed:\n" << cached.output;
            return 8;
        }

        std::cout << compile.output << cached.output;
        std::error_code ignored;
        std::filesystem::remove_all(cache, ignored);
        return 0;
    }

    int missing(char** argv) {
        const auto nonce = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()
        );
        const auto cache = std::filesystem::path(argv[2]) /
            ("missing-" + nonce);
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.string().c_str());
        environment("ORMI_CPU_JIT_CXX", "Z:/ormi/missing/compiler.exe");
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");

        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float32
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto output = ormi::core::ops::exp(square);

        try {
            output->realize();
        } catch (const ormi::jitError&) {
            std::error_code ignored;
            std::filesystem::remove_all(cache, ignored);
            return square->realized() || output->realized() ? 9 : 0;
        }

        std::cerr << "missing compiler did not fail on first fused use\n";
        std::error_code ignored;
        std::filesystem::remove_all(cache, ignored);
        return 10;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ormi_jit_compiler_helper MODE ...\n";
        return 2;
    }

    const std::string_view mode(argv[1]);

    try {
        if (mode == "verify" && argc == 5) {
            return verify(argv);
        }
        if (mode == "missing" && argc == 3) {
            return missing(argv);
        }
        if (
            (mode == "compile" && argc == 5) ||
            (mode == "cached" && argc == 3)
        ) {
            return work(argc, argv, mode);
        }
        std::cerr << "invalid JIT compiler helper arguments\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 6;
    }
}
