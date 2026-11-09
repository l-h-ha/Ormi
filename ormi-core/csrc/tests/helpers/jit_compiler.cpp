#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/jit.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/plan.h>
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
    bool execute(
        ormi::core::DType dtype,
        std::optional<std::size_t> expected_steps = std::nullopt,
        bool use_cosine = false
    ) {
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            dtype
        );
        auto* values = static_cast<T*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<T>(i) * static_cast<T>(0.5) - static_cast<T>(0.5);
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto summed = ormi::core::ops::add(square, input);
        const auto exponential = ormi::core::ops::exp(summed);
        const auto output = use_cosine
            ? ormi::core::ops::cos(exponential) : ormi::core::ops::sin(exponential);
        if (expected_steps) {
            ormi::core::Plan plan({output});
            if (output->realized() || plan.steps() != *expected_steps) {
                std::cerr << "expected " << *expected_steps << " lazy Plan steps, found "
                          << plan.steps() << '\n';
                return false;
            }
            plan.run();
        }
        const auto* result = static_cast<const T*>(output->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto exponential = std::exp(values[i] * values[i] + values[i]);
            const auto expected = use_cosine ? std::cos(exponential) : std::sin(exponential);
            const auto tolerance = sizeof(T) == sizeof(float)
                ? static_cast<T>(1.0e-6)
                : static_cast<T>(1.0e-12);
            if (!std::isfinite(result[i]) || std::abs(result[i] - expected) > tolerance) {
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
            std::cerr << "compile requires COMPILER and ID. The cached command does not.\n";
            return 2;
        }

        const std::optional<std::size_t> expected_steps = mode == "cached"
            ? std::optional<std::size_t>(1) : std::nullopt;
        if (
            !execute<float>(ormi::core::DType::float32, expected_steps) ||
            !execute<double>(ormi::core::DType::float64, expected_steps)
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

    int published_later(char** argv, bool existing) {
        const auto nonce = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()
        );
        const auto cache = std::filesystem::path(argv[2]) / ("later-" + nonce);
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.string().c_str());
        environment("ORMI_CPU_JIT_CXX", "Z:/ormi/missing/compiler.exe");
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");
        if (existing) {
            std::filesystem::create_directories(cache);
        }
        if (!execute<float>(ormi::core::DType::float32, 4) ||
            !execute<double>(ormi::core::DType::float64, 4)) {
            return 10;
        }

        const auto executable = std::filesystem::absolute(argv[0]);
        const auto compiled = ormi::platform::run(
            executable, std::vector<std::string>{"compile", cache.string(), argv[3], argv[4]}
        );
        if (compiled.state != ormi::platform::Process::State::exited || compiled.code != 0) {
            std::cerr << "publisher failed:\n" << compiled.output;
            return 11;
        }

        // A copied artifact can arrive without its writer's lock file.
        std::vector<std::filesystem::path> locks;
        for (const auto& entry : std::filesystem::directory_iterator(cache)) {
            if (entry.path().extension() == ".lock") {
                locks.push_back(entry.path());
            }
        }
        for (const auto& lock : locks) {
            std::filesystem::remove(lock);
        }
        if (!execute<float>(ormi::core::DType::float32, 1) ||
            !execute<double>(ormi::core::DType::float64, 1)) {
            return 12;
        }
        const bool use_cosine = true;
        if (!execute<float>(ormi::core::DType::float32, 4, use_cosine) ||
            !execute<double>(ormi::core::DType::float64, 4, use_cosine)) {
            return 14;
        }

        const auto empty = cache / "empty";
        environment("ORMI_CPU_JIT_CACHE_DIR", empty.string().c_str());
        if (!execute<float>(ormi::core::DType::float32, 1) ||
            !execute<double>(ormi::core::DType::float64, 1) || std::filesystem::exists(empty)) {
            return 13;
        }
        std::cout << "late publication loaded. Copied and in-memory kernels remain usable.\n";
        std::error_code ignored;
        std::filesystem::remove_all(cache, ignored);
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

    int missing(char** argv, bool automatic) {
        const auto nonce = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()
        );
        const auto cache = std::filesystem::path(argv[2]) /
            ("missing-" + nonce);
        environment("ORMI_CPU_JIT_CACHE_DIR", cache.string().c_str());
        environment("ORMI_CPU_JIT_CXX", automatic ? "" : "Z:/ormi/missing/compiler.exe");
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");
        if (automatic) {
            environment("PATH", "");
        }

        const bool correct =
            execute<float>(ormi::core::DType::float32) &&
            execute<double>(ormi::core::DType::float64);
        const auto entries = ormi::core::jit::info().entries;
        std::error_code ignored;
        std::filesystem::remove_all(cache, ignored);
        if (!correct || entries != 0) {
            std::cerr << "compiler-free execution failed or created JIT cache entries\n";
            return 9;
        }
        std::cout << "compiler-free CPU execution succeeded (entries=0)\n";
        return 0;
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
        if ((mode == "publish-missing" || mode == "publish-empty") && argc == 5) {
            return published_later(argv, mode == "publish-empty");
        }
        if ((mode == "missing" || mode == "absent") && argc == 3) {
            return missing(argv, mode == "absent");
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
