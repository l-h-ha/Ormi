#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/compiler.h"
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

    class Cache {
    public:
        Cache()
            : path_(
                std::filesystem::temp_directory_path() /
                ("ormi-jit-kernel-test-" +
                    std::to_string(ormi::platform::process_id()))
            )
        {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
            environment("ORMI_CPU_JIT_CACHE_DIR", path_.string().c_str());
        }

        ~Cache() {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
        }

    private:
        std::filesystem::path path_;
    };

    template <typename T>
    std::shared_ptr<ormi::core::Tensor> tensor(ormi::core::DType dtype) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            dtype
        );
        auto* data = static_cast<T*>(result->data());
        data[0] = static_cast<T>(-0.5);
        data[1] = static_cast<T>(0.0);
        data[2] = static_cast<T>(0.5);
        data[3] = static_cast<T>(1.0);
        return result;
    }

    TEST(CPUJITKernelTest, executesFloat32FusedChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        const auto input = tensor<float>(ormi::core::DType::float32);
        const auto square = ormi::core::ops::mul(input, input);
        const auto sum = ormi::core::ops::add(square, input);
        const auto exponential = ormi::core::ops::exp(sum);
        const auto output = ormi::core::ops::sin(exponential);

        const auto* data = static_cast<const float*>(output->data());

        const float values[]{-0.5f, 0.0f, 0.5f, 1.0f};
        for (std::size_t i = 0; i < output->numel(); ++i) {
            EXPECT_FLOAT_EQ(data[i], std::sin(std::exp(values[i] * values[i] + values[i])));
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(sum->realized());
        EXPECT_FALSE(exponential->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST(CPUJITKernelTest, executesFloat64FusedChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        const auto input = tensor<double>(ormi::core::DType::float64);
        const auto square = ormi::core::ops::mul(input, input);
        const auto sum = ormi::core::ops::add(square, input);
        const auto exponential = ormi::core::ops::exp(sum);
        const auto output = ormi::core::ops::cos(exponential);

        const auto* data = static_cast<const double*>(output->data());

        const double values[]{-0.5, 0.0, 0.5, 1.0};
        for (std::size_t i = 0; i < output->numel(); ++i) {
            EXPECT_DOUBLE_EQ(data[i], std::cos(std::exp(values[i] * values[i] + values[i])));
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(sum->realized());
        EXPECT_FALSE(exponential->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST(CPUJITKernelTest, singleOperationDoesNotResolveCompiler) {
        environment("ORMI_CPU_JIT_CXX", "Z:/ormi/missing/compiler.exe");
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");
        const auto input = tensor<float>(ormi::core::DType::float32);
        const auto output = ormi::core::ops::neg(input);

        const auto* data = static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(data[0], 0.5f);
        EXPECT_TRUE(output->realized());
    }

    TEST(CPUJITKernelTest, missingCompilerFailsLazily) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "Z:/ormi/missing/compiler.exe");
        environment("ORMI_CPU_JIT_CXX_ID", "gcc");
        const auto input = tensor<float>(ormi::core::DType::float32);
        const auto square = ormi::core::ops::mul(input, input);
        const auto output = ormi::core::ops::exp(square);

        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(output->realized());
        EXPECT_THROW(output->realize(), ormi::jitError);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(output->realized());
    }

#if defined(_WIN32)
    TEST(CPUJITKernelTest, executesWithConfiguredMSVCOutsideDeveloperShell) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "msvc");

        const auto input = tensor<float>(ormi::core::DType::float32);
        const auto square = ormi::core::ops::mul(input, input);
        const auto output = ormi::core::ops::exp(square);
        const float* data = nullptr;

        try {
            data = static_cast<const float*>(output->data());
        } catch (const ormi::jitError& error) {
            GTEST_SKIP() << "Configured MSVC is unavailable: " << error.what();
        }

        EXPECT_FLOAT_EQ(data[0], std::exp(0.25f));
        EXPECT_FLOAT_EQ(data[1], 1.0f);
        EXPECT_FLOAT_EQ(data[2], std::exp(0.25f));
        EXPECT_FLOAT_EQ(data[3], std::exp(1.0f));
        EXPECT_FALSE(square->realized());
    }
#endif
}
