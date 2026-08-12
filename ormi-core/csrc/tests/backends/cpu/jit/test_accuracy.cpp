#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
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

    class Cache {
    public:
        Cache()
            : path_(
                std::filesystem::temp_directory_path() /
                ("ormi-jit-accuracy-test-" +
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

    std::shared_ptr<ormi::core::Tensor> constant(ormi::core::Shape shape) {
        auto result = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(result->data()),
            result->numel(),
            0.5f
        );
        return result;
    }

    std::shared_ptr<ormi::core::Tensor> mapped_sum(
        const std::shared_ptr<ormi::core::Tensor>& input
    ) {
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        return ormi::core::ops::sum(ormi::core::ops::sin(added));
    }

    double relative_error_ppm(float actual, double expected) {
        return std::abs(static_cast<double>(actual) - expected) /
            std::abs(expected) * 1.0e6;
    }

    TEST(CPUJITAccuracyTest, boundsContiguousMappedReductionError) {
        Cache cache;
        constexpr std::size_t elements = 1024 * 1024;
        const auto input = constant({static_cast<std::int64_t>(elements)});
        auto* values = static_cast<float*>(input->data());
        double expected = 0.0;
        for (std::size_t i = 0; i < elements; ++i) {
            const auto value =
                static_cast<float>(static_cast<int>(i % 257) - 128) /
                256.0f;
            values[i] = value;
            const float square = value * value;
            const float added = square + value;
            expected += static_cast<double>(std::sin(added));
        }
        const auto output = mapped_sum(input);

        const auto actual = *static_cast<const float*>(output->data());

        EXPECT_LE(relative_error_ppm(actual, expected), 0.5);
    }

    TEST(CPUJITAccuracyTest, boundsTransposedMappedReductionError) {
        Cache cache;
        constexpr std::int64_t extent = 1024;
        constexpr std::size_t elements = extent * extent;
        const auto base = constant({extent, extent});
        const auto input = ormi::core::ops::transpose(base, {1, 0});
        const auto output = mapped_sum(input);

        const auto actual = *static_cast<const float*>(output->data());
        const auto expected =
            static_cast<double>(std::sin(0.75f)) *
            static_cast<double>(elements);

        EXPECT_LE(relative_error_ppm(actual, expected), 0.5);
    }

    TEST(CPUJITAccuracyTest, boundsSineAndCosinePointwiseError) {
        Cache cache;
        constexpr std::size_t elements = 32 * 1024;
        const auto input = constant({static_cast<std::int64_t>(elements)});
        const auto scale = constant({static_cast<std::int64_t>(elements)});
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < elements; ++i) {
            values[i] =
                static_cast<float>(static_cast<int>(i % 20001) - 10000) /
                10.0f;
        }
        values[0] = 0.0f;
        values[1] = -0.0f;
        values[2] = 1.0e10f;
        values[3] = -1.0e10f;
        values[4] = std::numeric_limits<float>::infinity();
        values[5] = -std::numeric_limits<float>::infinity();
        values[6] = std::numeric_limits<float>::quiet_NaN();
        const auto sine = ormi::core::ops::mul(
            ormi::core::ops::sin(input),
            scale
        );
        const auto cosine = ormi::core::ops::mul(
            ormi::core::ops::cos(input),
            scale
        );

        const auto* actual_sine = static_cast<const float*>(sine->data());
        const auto* actual_cosine = static_cast<const float*>(cosine->data());
        for (std::size_t i = 0; i < elements; ++i) {
            const auto expected_sine = std::sin(values[i]) * 0.5f;
            const auto expected_cosine = std::cos(values[i]) * 0.5f;
            if (std::isnan(expected_sine)) {
                EXPECT_TRUE(std::isnan(actual_sine[i])) << "sine index " << i;
            } else {
                EXPECT_NEAR(actual_sine[i], expected_sine, 2.0e-6f)
                    << "sine index " << i;
            }
            if (std::isnan(expected_cosine)) {
                EXPECT_TRUE(std::isnan(actual_cosine[i]))
                    << "cosine index " << i;
            } else {
                EXPECT_NEAR(actual_cosine[i], expected_cosine, 2.0e-6f)
                    << "cosine index " << i;
            }
        }
    }

    TEST(CPUJITAccuracyTest, boundsFusedVectorMathErrorAndScalarTail) {
        Cache cache;
        constexpr std::size_t elements = 32 * 1024 + 3;
        const auto input = constant({static_cast<std::int64_t>(elements)});
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < elements; ++i) {
            values[i] =
                static_cast<float>(static_cast<int>(i % 1801) - 900) /
                1000.0f;
        }
        const auto output = ormi::core::ops::cos(
            ormi::core::ops::sin(
                ormi::core::ops::log(
                    ormi::core::ops::exp(input)
                )
            )
        );

        const auto* actual = static_cast<const float*>(output->data());
        for (std::size_t i = 0; i < elements; ++i) {
            const auto exponential = std::exp(values[i]);
            const auto logarithm = std::log(exponential);
            const auto sine = std::sin(logarithm);
            const auto expected = std::cos(sine);
            EXPECT_NEAR(actual[i], expected, 3.0e-6f) << "index " << i;
        }
    }

    TEST(CPUJITAccuracyTest, boundsVectorizedMappedReductionError) {
        Cache cache;
        constexpr std::size_t elements = 4 * 1024 * 1024;
        const auto input = constant({static_cast<std::int64_t>(elements)});
        std::fill_n(
            static_cast<float*>(input->data()),
            input->numel(),
            0.1f
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(
            ormi::core::ops::mul(added, input)
        );

        const auto actual = *static_cast<const float*>(output->data());
        const float value = 0.1f;
        const float mapped = (value * value + value) * value;
        const auto expected =
            static_cast<double>(mapped) * static_cast<double>(elements);

        EXPECT_LE(relative_error_ppm(actual, expected), 0.5);
    }

    TEST(CPUJITAccuracyTest, boundsVectorizedTransposedReductionError) {
        Cache cache;
        constexpr std::int64_t extent = 1024;
        constexpr std::size_t elements = extent * extent;
        const auto base = constant({extent, extent});
        std::fill_n(
            static_cast<float*>(base->data()),
            base->numel(),
            0.1f
        );
        const auto input = ormi::core::ops::transpose(base, {1, 0});
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(
            ormi::core::ops::mul(added, input)
        );

        const auto actual = *static_cast<const float*>(output->data());
        const float value = 0.1f;
        const float mapped = (value * value + value) * value;
        const auto expected =
            static_cast<double>(mapped) * static_cast<double>(elements);

        EXPECT_LE(relative_error_ppm(actual, expected), 0.5);
    }

    TEST(CPUJITAccuracyTest, boundsPackedAutocastReductionError) {
        Cache cache;
        constexpr std::size_t elements = 4 * 1024 * 1024;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{static_cast<std::int64_t>(elements)},
            ormi::core::DType::float16
        );
        const ormi::core::Float16 value{0.1f};
        std::fill_n(
            static_cast<ormi::core::Float16*>(input->data()),
            input->numel(),
            value
        );
        std::shared_ptr<ormi::core::Tensor> output;
        ormi::core::Float16 mapped;
        {
            const ormi::core::Autocast autocast;
            const auto square = ormi::core::ops::mul(input, input);
            const auto added = ormi::core::ops::add(square, input);
            output = ormi::core::ops::sum(
                ormi::core::ops::mul(added, input)
            );
            const ormi::core::Float16 square_value{
                static_cast<float>(value) * static_cast<float>(value)
            };
            const ormi::core::Float16 added_value{
                static_cast<float>(square_value) + static_cast<float>(value)
            };
            mapped = ormi::core::Float16{
                static_cast<float>(added_value) * static_cast<float>(value)
            };
        }

        const auto actual = *static_cast<const float*>(output->data());
        const auto expected = static_cast<double>(static_cast<float>(mapped)) *
            static_cast<double>(elements);

        EXPECT_LE(relative_error_ppm(actual, expected), 0.5);
    }
}
