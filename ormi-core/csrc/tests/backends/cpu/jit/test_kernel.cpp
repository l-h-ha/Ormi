#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/compiler.h"
#include "backends/cpu/jit/kernel.h"
#include "core/runtime/fusion/expr.h"
#include "core/runtime/step.h"
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

    TEST(CPUJITKernelTest, executesPreparedKernelWithAuxiliaryOutput) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        const auto input = tensor<float>(ormi::core::DType::float32);
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto sine = ormi::core::ops::sin(mapped);
        const auto cosine = ormi::core::ops::cos(mapped);
        const std::array<ormi::core::Tensor*, 1> auxiliaries{cosine.get()};
        const std::unordered_set<const ormi::core::Tensor*> boundaries;
        auto expression = ormi::core::runtime::fusion::build(
            *sine,
            auxiliaries,
            boundaries
        );
        ASSERT_TRUE(expression.has_value());

        auto step = ormi::core::runtime::Step::deferred_fusion(
            *sine,
            static_cast<ormi::core::runtime::PrepareFusionFn>(
                ormi::backends::cpu::jit::prepare
            ),
            nullptr,
            static_cast<ormi::core::runtime::FusionFn>(
                ormi::backends::cpu::jit::execute
            ),
            std::move(*expression),
            {cosine.get()}
        );
        step.prepare();

        EXPECT_EQ(step.output_count(), 2u);
        EXPECT_EQ(&step.output(0), sine.get());
        EXPECT_EQ(&step.output(1), cosine.get());
        step.run();

        const auto* actual_sine = static_cast<const float*>(sine->data());
        const auto* actual_cosine = static_cast<const float*>(cosine->data());
        const float values[]{-0.5f, 0.0f, 0.5f, 1.0f};
        for (std::size_t i = 0; i < sine->numel(); ++i) {
            const auto mapped_value = values[i] * values[i] + values[i];
            EXPECT_FLOAT_EQ(actual_sine[i], std::sin(mapped_value));
            EXPECT_FLOAT_EQ(actual_cosine[i], std::cos(mapped_value));
        }
        EXPECT_TRUE(sine->realized());
        EXPECT_TRUE(cosine->realized());
        EXPECT_FALSE(mapped->realized());
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

    TEST(CPUJITKernelTest, executesVectorizedArithmeticChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1024},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(i % 17) * 0.125f;
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mul(added, input);
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            EXPECT_FLOAT_EQ(result[i], (values[i] * values[i] + values[i]) * values[i]);
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesLargeArithmeticChainAcrossRanges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t elements = 256 * 1024 + 17;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{static_cast<std::int64_t>(elements)},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < elements; ++i) {
            values[i] = static_cast<float>(i % 17) * 0.125f;
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mul(added, input);
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < elements; ++i) {
            const auto expected =
                (values[i] * values[i] + values[i]) * values[i];
            if (result[i] != expected) {
                ADD_FAILURE()
                    << "Incorrect result at index " << i
                    << ": expected " << expected
                    << ", received " << result[i];
                break;
            }
        }
    }

    TEST(CPUJITKernelTest, executesVectorizedScalarBroadcastChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1024},
            ormi::core::DType::float32
        );
        auto scale = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float32
        );
        auto bias = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        *static_cast<float*>(scale->data()) = 1.5f;
        *static_cast<float*>(bias->data()) = -0.25f;
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(i % 17) * 0.125f;
        }

        const auto scaled = ormi::core::ops::mul(input, scale);
        const auto shifted = ormi::core::ops::add(scaled, bias);
        const auto output = ormi::core::ops::mul(shifted, input);
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            EXPECT_FLOAT_EQ(result[i], (values[i] * 1.5f - 0.25f) * values[i]);
        }
        EXPECT_FALSE(scaled->realized());
        EXPECT_FALSE(shifted->realized());
    }

    TEST(CPUJITKernelTest, executesVectorizedFloat64ScalarBroadcastChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1024},
            ormi::core::DType::float64
        );
        auto scale = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float64
        );
        auto* values = static_cast<double*>(input->data());
        *static_cast<double*>(scale->data()) = 0.25;
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<double>(i % 17) * 0.125;
        }

        const auto scaled = ormi::core::ops::mul(input, scale);
        const auto shifted = ormi::core::ops::add(scaled, scale);
        const auto output = ormi::core::ops::mul(shifted, input);
        const auto* result = static_cast<const double*>(output->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            EXPECT_DOUBLE_EQ(result[i], (values[i] * 0.25 + 0.25) * values[i]);
        }
        EXPECT_FALSE(scaled->realized());
        EXPECT_FALSE(shifted->realized());
    }

    TEST(CPUJITKernelTest, executesVectorizedRowBroadcastChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 4},
            ormi::core::DType::float32
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float32
        );
        auto* lhs_values = static_cast<float*>(lhs->data());
        auto* rhs_values = static_cast<float*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = static_cast<float>(i + 1) * 0.25f;
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            rhs_values[i] = static_cast<float>(i + 1) * 0.5f;
        }

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            EXPECT_FLOAT_EQ(
                result[i],
                (lhs_values[i] * lhs_values[i] + rhs_values[i % 4]) * lhs_values[i]
            );
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesVectorizedColumnBroadcastChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 4},
            ormi::core::DType::float64
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float64
        );
        auto* lhs_values = static_cast<double*>(lhs->data());
        auto* rhs_values = static_cast<double*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = static_cast<double>(i + 1) * 0.25;
        }
        rhs_values[0] = 0.5;
        rhs_values[1] = 1.5;

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto* result = static_cast<const double*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            EXPECT_DOUBLE_EQ(
                result[i],
                (lhs_values[i] * lhs_values[i] + rhs_values[i / 4]) * lhs_values[i]
            );
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesLargeMixedBroadcastAcrossRanges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1024, 513},
            ormi::core::DType::float32
        );
        auto row = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{513},
            ormi::core::DType::float32
        );
        auto column = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1024, 1},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        auto* row_values = static_cast<float*>(row->data());
        auto* column_values = static_cast<float*>(column->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(i % 17) * 0.125f;
        }
        for (std::size_t i = 0; i < row->numel(); ++i) {
            row_values[i] = static_cast<float>(i % 11) * 0.0625f;
        }
        for (std::size_t i = 0; i < column->numel(); ++i) {
            column_values[i] = static_cast<float>(i % 7) * 0.25f;
        }

        const auto scaled = ormi::core::ops::mul(input, row);
        const auto shifted = ormi::core::ops::add(scaled, column);
        const auto output = ormi::core::ops::mul(shifted, input);
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto expected =
                (values[i] * row_values[i % 513] +
                    column_values[i / 513]) *
                values[i];
            if (result[i] != expected) {
                ADD_FAILURE()
                    << "Incorrect result at index " << i
                    << ": expected " << expected
                    << ", received " << result[i];
                break;
            }
        }
        EXPECT_FALSE(scaled->realized());
        EXPECT_FALSE(shifted->realized());
    }

    TEST(CPUJITKernelTest, executesLargeFloat16BroadcastAcrossRanges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1024, 513},
            ormi::core::DType::float16
        );
        auto row = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{513},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        auto* row_values = static_cast<Float16*>(row->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{static_cast<float>(i % 17) * 0.03125f};
        }
        for (std::size_t i = 0; i < row->numel(); ++i) {
            row_values[i] =
                Float16{static_cast<float>(i % 11) * 0.015625f};
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, row);
        const auto output = ormi::core::ops::mul(added, input);
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < input->numel(); ++i) {
            const auto expected =
                (values[i] * values[i] + row_values[i % 513]) *
                values[i];
            if (result[i].bits() != expected.bits()) {
                ADD_FAILURE()
                    << "Incorrect Float16 result at index " << i;
                break;
            }
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesVectorizedArbitraryRankBroadcastChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3, 4},
            ormi::core::DType::float32
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3, 4},
            ormi::core::DType::float32
        );
        auto* lhs_values = static_cast<float*>(lhs->data());
        auto* rhs_values = static_cast<float*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = static_cast<float>(i + 1) * 0.125f;
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            rhs_values[i] = static_cast<float>(i + 1) * 0.25f;
        }

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            EXPECT_FLOAT_EQ(
                result[i],
                (lhs_values[i] * lhs_values[i] + rhs_values[i % 12]) * lhs_values[i]
            );
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesMixedArbitraryRankBroadcastChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1, 4},
            ormi::core::DType::float32
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{1, 3, 1},
            ormi::core::DType::float32
        );
        auto* lhs_values = static_cast<float*>(lhs->data());
        auto* rhs_values = static_cast<float*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = static_cast<float>(i + 1) * 0.25f;
        }
        rhs_values[0] = 0.5f;
        rhs_values[1] = 1.0f;
        rhs_values[2] = 1.5f;

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t batch = 0; batch < 2; ++batch) {
            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t column = 0; column < 4; ++column) {
                    const auto output_index = batch * 12 + row * 4 + column;
                    const auto lhs_value = lhs_values[batch * 4 + column];
                    EXPECT_FLOAT_EQ(
                        result[output_index],
                        (lhs_value * lhs_value + rhs_values[row]) * lhs_value
                    );
                }
            }
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesEmptyVectorizedBroadcastChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{0, 4},
            ormi::core::DType::float32
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float32
        );
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);

        output->realize();

        EXPECT_EQ(output->numel(), 0u);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesBroadcastedFloat32Chain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float32
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3},
            ormi::core::DType::float32
        );
        auto* lhs_data = static_cast<float*>(lhs->data());
        auto* rhs_data = static_cast<float*>(rhs->data());
        lhs_data[0] = 2.0f;
        lhs_data[1] = 4.0f;
        rhs_data[0] = 1.0f;
        rhs_data[1] = 2.0f;
        rhs_data[2] = 3.0f;

        const auto sum = ormi::core::ops::add(lhs, rhs);
        const auto output = ormi::core::ops::sin(ormi::core::ops::mul(sum, lhs));
        const auto* data = static_cast<const float*>(output->data());

        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                const auto expected = std::sin(
                    (lhs_data[row] + rhs_data[column]) * lhs_data[row]
                );
                EXPECT_FLOAT_EQ(data[row * 3 + column], expected);
            }
        }
        EXPECT_FALSE(sum->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST(CPUJITKernelTest, executesEmptyBroadcastedChain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{0, 3},
            ormi::core::DType::float32
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3},
            ormi::core::DType::float32
        );
        const auto sum = ormi::core::ops::add(lhs, rhs);
        const auto output = ormi::core::ops::sin(sum);

        output->realize();

        EXPECT_EQ(output->numel(), 0u);
        EXPECT_FALSE(sum->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST(CPUJITKernelTest, executesTransposedFloat32Chain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3},
            ormi::core::DType::float32
        );
        auto* base_data = static_cast<float*>(base->data());
        for (std::size_t i = 0; i < base->numel(); ++i) {
            base_data[i] = static_cast<float>(i + 1);
        }

        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto square = ormi::core::ops::mul(transposed, transposed);
        const auto output = ormi::core::ops::sin(ormi::core::ops::add(square, transposed));
        const auto* data = static_cast<const float*>(output->data());
        const float logical[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};

        for (std::size_t i = 0; i < output->numel(); ++i) {
            EXPECT_FLOAT_EQ(data[i], std::sin(logical[i] * logical[i] + logical[i]));
        }
        EXPECT_FALSE(square->realized());
    }

    TEST(CPUJITKernelTest, executesRankThreePermutationWithLinearInput) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3, 4},
            ormi::core::DType::float32
        );
        auto linear = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4, 2, 3},
            ormi::core::DType::float32
        );
        auto* base_data = static_cast<float*>(base->data());
        auto* linear_data = static_cast<float*>(linear->data());
        for (std::size_t i = 0; i < base->numel(); ++i) {
            base_data[i] = static_cast<float>(i + 1);
            linear_data[i] = static_cast<float>(i) * 0.25f;
        }

        const auto permuted = ormi::core::ops::transpose(base, {2, 0, 1});
        const auto square = ormi::core::ops::mul(permuted, permuted);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(square, linear)
        );
        const auto* data = static_cast<const float*>(output->data());

        for (std::size_t axis0 = 0; axis0 < 4; ++axis0) {
            for (std::size_t axis1 = 0; axis1 < 2; ++axis1) {
                for (std::size_t axis2 = 0; axis2 < 3; ++axis2) {
                    const auto output_index =
                        axis0 * 6 + axis1 * 3 + axis2;
                    const auto input_index =
                        axis1 * 12 + axis2 * 4 + axis0;
                    const auto value = base_data[input_index];
                    EXPECT_FLOAT_EQ(
                        data[output_index],
                        std::sin(
                            value * value + linear_data[output_index]
                        )
                    );
                }
            }
        }
        EXPECT_FALSE(square->realized());
    }

    TEST(CPUJITKernelTest, executesTiledTransposeWithPartialEdges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{35, 33},
            ormi::core::DType::float32
        );
        auto* base_data = static_cast<float*>(base->data());
        for (std::size_t i = 0; i < base->numel(); ++i) {
            base_data[i] = static_cast<float>(i) * 0.001f;
        }

        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto sum = ormi::core::ops::add(transposed, transposed);
        const auto output = ormi::core::ops::sin(sum);
        const auto* data = static_cast<const float*>(output->data());

        for (std::size_t row = 0; row < 33; ++row) {
            for (std::size_t column = 0; column < 35; ++column) {
                const auto output_index = row * 35 + column;
                const auto input_index = column * 33 + row;
                EXPECT_FLOAT_EQ(
                    data[output_index],
                    std::sin(base_data[input_index] * 2.0f)
                );
            }
        }
        EXPECT_FALSE(sum->realized());
    }

    TEST(CPUJITKernelTest, executesExpandedFloat32Chain) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float32
        );
        auto* base_data = static_cast<float*>(base->data());
        base_data[0] = 2.0f;
        base_data[1] = 4.0f;

        const auto expanded = ormi::core::ops::expand(base, {2, 3});
        const auto sum = ormi::core::ops::add(expanded, expanded);
        const auto output = ormi::core::ops::sin(sum);
        const auto* data = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_FLOAT_EQ(data[i], std::sin(4.0f));
            EXPECT_FLOAT_EQ(data[i + 3], std::sin(8.0f));
        }
        EXPECT_FALSE(sum->realized());
    }

    TEST(CPUJITKernelTest, executesFloat16ChainWithNodeRounding) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float16
        );
        auto* input_data = static_cast<Float16*>(input->data());
        input_data[0] = Float16{-0.5f};
        input_data[1] = Float16{0.1f};
        input_data[2] = Float16{0.5f};
        input_data[3] = Float16{1.0f};

        const auto square = ormi::core::ops::mul(input, input);
        const auto sum = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sin(sum);
        const auto* data = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto rounded_square = input_data[i] * input_data[i];
            const auto rounded_sum = rounded_square + input_data[i];
            const auto expected = Float16{std::sin(static_cast<float>(rounded_sum))};
            EXPECT_EQ(data[i].bits(), expected.bits());
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(sum->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16ArithmeticExactly) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f};
        }
        values[0] = Float16::from_bits(0x7d01u);
        values[1] = Float16::from_bits(0xfd01u);
        values[2] = Float16{std::numeric_limits<float>::infinity()};
        values[3] = Float16{-std::numeric_limits<float>::infinity()};
        values[4] = Float16::from_bits(0x0001u);

        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mul(added, input);
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected = (values[i] * values[i] + values[i]) * values[i];
            EXPECT_EQ(result[i].bits(), expected.bits()) << "index " << i;
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16ScalarBroadcastExactly) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto scale = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{
                static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f
            };
        }
        const auto factor = Float16{0.3333f};
        *static_cast<Float16*>(scale->data()) = factor;

        const auto scaled = ormi::core::ops::mul(input, scale);
        const auto shifted = ormi::core::ops::add(scaled, scale);
        const auto output = ormi::core::ops::mul(shifted, input);
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected = (values[i] * factor + factor) * values[i];
            EXPECT_EQ(result[i].bits(), expected.bits()) << "index " << i;
        }
        EXPECT_FALSE(scaled->realized());
        EXPECT_FALSE(shifted->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16RowBroadcastWithTail) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4, 17},
            ormi::core::DType::float16
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{17},
            ormi::core::DType::float16
        );
        auto* lhs_values = static_cast<Float16*>(lhs->data());
        auto* rhs_values = static_cast<Float16*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = Float16{
                static_cast<float>(static_cast<int>(i % 19) - 9) * 0.03125f
            };
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            rhs_values[i] = Float16{static_cast<float>(i + 1) * 0.015625f};
        }

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected =
                (lhs_values[i] * lhs_values[i] + rhs_values[i % 17]) * lhs_values[i];
            EXPECT_EQ(result[i].bits(), expected.bits()) << "index " << i;
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16ColumnBroadcastWithTail) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4, 17},
            ormi::core::DType::float16
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4, 1},
            ormi::core::DType::float16
        );
        auto* lhs_values = static_cast<Float16*>(lhs->data());
        auto* rhs_values = static_cast<Float16*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = Float16{static_cast<float>(i % 17) * 0.03125f};
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            rhs_values[i] = Float16{static_cast<float>(i + 1) * 0.125f};
        }

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected =
                (lhs_values[i] * lhs_values[i] + rhs_values[i / 17]) * lhs_values[i];
            EXPECT_EQ(result[i].bits(), expected.bits()) << "index " << i;
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16RepeatedSpanBroadcast) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3, 17},
            ormi::core::DType::float16
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{3, 17},
            ormi::core::DType::float16
        );
        auto* lhs_values = static_cast<Float16*>(lhs->data());
        auto* rhs_values = static_cast<Float16*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = Float16{static_cast<float>(i % 23) * 0.015625f};
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            rhs_values[i] = Float16{static_cast<float>(i % 13) * 0.03125f};
        }

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected =
                (lhs_values[i] * lhs_values[i] + rhs_values[i % 51]) * lhs_values[i];
            EXPECT_EQ(result[i].bits(), expected.bits()) << "index " << i;
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesFloat16RowBroadcastWithBaselineTarget) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "baseline");
        using ormi::core::Float16;
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4, 17},
            ormi::core::DType::float16
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{17},
            ormi::core::DType::float16
        );
        auto* lhs_values = static_cast<Float16*>(lhs->data());
        auto* rhs_values = static_cast<Float16*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = Float16{static_cast<float>(i % 17) * 0.03125f};
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            rhs_values[i] = Float16{static_cast<float>(i + 1) * 0.015625f};
        }

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected =
                (lhs_values[i] * lhs_values[i] + rhs_values[i % 17]) * lhs_values[i];
            EXPECT_EQ(result[i].bits(), expected.bits());
        }
    }

    TEST(CPUJITKernelTest, executesFloat16ArithmeticWithBaselineTarget) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "baseline");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{static_cast<float>(i) * 0.03125f};
        }
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mul(added, input);
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected = (values[i] * values[i] + values[i]) * values[i];
            EXPECT_EQ(result[i].bits(), expected.bits());
        }
    }

    TEST(CPUJITKernelTest, executesPackedFloat16GlobalSumWithTail) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{
                static_cast<float>(static_cast<int>(i % 17) - 8) * 0.03125f
            };
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto mapped = ormi::core::ops::mul(added, input);
        const auto output = ormi::core::ops::sum(mapped);
        const auto result = *static_cast<const Float16*>(output->data());

        float expected = 0.0f;
        for (const auto value : std::span<const Float16>(values, input->numel())) {
            expected += static_cast<float>((value * value + value) * value);
        }
        EXPECT_EQ(result.bits(), Float16{expected}.bits());
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16LastAxisMeanWithTail) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 65},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{
                static_cast<float>(static_cast<int>(i % 13) - 6) * 0.03125f
            };
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t row = 0; row < 2; ++row) {
            float expected = 0.0f;
            for (std::size_t column = 0; column < 65; ++column) {
                const auto value = values[row * 65 + column];
                expected += static_cast<float>(value * value + value);
            }
            EXPECT_EQ(result[row].bits(), Float16{expected / 65.0f}.bits());
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesTiledFloat16FirstAxisMeanWithTail) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{5, 70},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{
                static_cast<float>(static_cast<int>(i % 11) - 5) *
                    0.03125f
            };
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped, {0});
        const auto* result = static_cast<const Float16*>(output->data());

        for (std::size_t column = 0; column < 70; ++column) {
            float expected = 0.0f;
            for (std::size_t row = 0; row < 5; ++row) {
                const auto value = values[row * 70 + column];
                expected += static_cast<float>(value * value + value);
            }
            EXPECT_EQ(
                result[column].bits(),
                Float16{expected / 5.0f}.bits()
            );
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesLargeTiledFirstAxisSumAcrossRanges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 513;
        constexpr std::size_t columns = 1025;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{rows, columns},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                values[row * columns + column] =
                    static_cast<float>(
                        static_cast<int>((row + column) % 7) - 3
                    ) * 0.03125f;
            }
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {0});
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t column = 0; column < columns; ++column) {
            float expected = 0.0f;
            for (std::size_t row = 0; row < rows; ++row) {
                const auto value = values[row * columns + column];
                expected += value * value + value;
            }
            EXPECT_NEAR(result[column], expected, 1.0e-5f);
        }
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesLargeGlobalMeanAcrossPartialRanges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t elements = 524291;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        std::fill_n(values, elements, 0.25f);

        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped);
        const auto result = *static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(result, 0.3125f);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, usesFloat32PartialsForLargeFloat16Mean) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t elements = 524291;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{elements},
            ormi::core::DType::float16
        );
        auto* values = static_cast<ormi::core::Float16*>(input->data());
        std::fill_n(values, elements, ormi::core::Float16{0.25f});

        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped);
        const auto result =
            *static_cast<const ormi::core::Float16*>(output->data());

        EXPECT_EQ(result.bits(), ormi::core::Float16{0.3125f}.bits());
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesLargeTransposedGlobalSumAcrossPartialSlabs) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 513;
        constexpr std::size_t columns = 1025;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{columns, rows},
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(base->data()),
            base->numel(),
            0.25f
        );

        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto square =
            ormi::core::ops::mul(transposed, transposed);
        const auto added =
            ormi::core::ops::add(square, transposed);
        const auto mapped =
            ormi::core::ops::mul(added, transposed);
        const auto output = ormi::core::ops::sum(mapped);
        const auto result = *static_cast<const float*>(output->data());
        const auto expected =
            static_cast<float>(rows * columns) * 0.078125f;

        EXPECT_FLOAT_EQ(result, expected);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesVectorizedFloat64TransposeMeanWithTail) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 513;
        constexpr std::size_t columns = 1025;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{columns, rows},
            ormi::core::DType::float64
        );
        std::fill_n(
            static_cast<double*>(base->data()),
            base->numel(),
            0.25
        );

        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto square =
            ormi::core::ops::mul(transposed, transposed);
        const auto mapped =
            ormi::core::ops::add(square, transposed);
        const auto output = ormi::core::ops::mean(mapped);
        const auto result = *static_cast<const double*>(output->data());

        EXPECT_DOUBLE_EQ(result, 0.3125);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesRankThreeGlobalMeanAcrossPartialSlabs) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65, 33, 257},
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(base->data()),
            base->numel(),
            0.25f
        );

        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto square = ormi::core::ops::mul(permuted, permuted);
        const auto mapped = ormi::core::ops::add(square, permuted);
        const auto output = ormi::core::ops::mean(mapped);
        const auto result = *static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(result, 0.3125f);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesTiledRankThreeMiddleAxisSum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t first = 33;
        constexpr std::size_t middle = 65;
        constexpr std::size_t last = 97;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{middle, last, first},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(base->data());
        for (std::size_t j = 0; j < middle; ++j) {
            for (std::size_t k = 0; k < last; ++k) {
                for (std::size_t i = 0; i < first; ++i) {
                    values[(j * last + k) * first + i] =
                        static_cast<float>(i) * 0.01f +
                        static_cast<float>(k) * 0.001f;
                }
            }
        }

        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto square = ormi::core::ops::mul(permuted, permuted);
        const auto mapped = ormi::core::ops::add(square, permuted);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < first; ++i) {
            for (std::size_t k = 0; k < last; ++k) {
                const auto value =
                    static_cast<float>(i) * 0.01f +
                    static_cast<float>(k) * 0.001f;
                const auto expected =
                    static_cast<float>(middle) * (value * value + value);
                EXPECT_NEAR(result[i * last + k], expected, 1e-4f);
            }
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesTiledRankThreeMixedGlobalSum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t first = 33;
        constexpr std::size_t middle = 65;
        constexpr std::size_t last = 97;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{middle, last, first},
            ormi::core::DType::float32
        );
        auto contiguous = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{first, middle, last},
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(base->data()),
            base->numel(),
            0.25f
        );
        std::fill_n(
            static_cast<float*>(contiguous->data()),
            contiguous->numel(),
            0.5f
        );

        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto square = ormi::core::ops::mul(permuted, permuted);
        const auto mapped = ormi::core::ops::add(square, contiguous);
        const auto output = ormi::core::ops::sum(mapped);
        const auto result = *static_cast<const float*>(output->data());
        const auto expected =
            static_cast<float>(first * middle * last) * 0.5625f;

        EXPECT_FLOAT_EQ(result, expected);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, usesFloat32PartialsForExpandedFloat16Mean) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 513;
        constexpr std::size_t columns = 1025;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{rows, 1},
            ormi::core::DType::float16
        );
        std::fill_n(
            static_cast<ormi::core::Float16*>(base->data()),
            base->numel(),
            ormi::core::Float16{0.25f}
        );

        const auto expanded =
            ormi::core::ops::expand(base, {rows, columns});
        const auto square = ormi::core::ops::mul(expanded, expanded);
        const auto mapped = ormi::core::ops::add(square, expanded);
        const auto output = ormi::core::ops::mean(mapped);
        const auto result =
            *static_cast<const ormi::core::Float16*>(output->data());

        EXPECT_EQ(result.bits(), ormi::core::Float16{0.3125f}.bits());
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesMixedLayoutGlobalSumAcrossPartialSlabs) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 513;
        constexpr std::size_t columns = 1025;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{columns, rows},
            ormi::core::DType::float32
        );
        auto contiguous = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{rows, columns},
            ormi::core::DType::float32
        );
        std::fill_n(
            static_cast<float*>(base->data()),
            base->numel(),
            0.25f
        );
        std::fill_n(
            static_cast<float*>(contiguous->data()),
            contiguous->numel(),
            0.5f
        );

        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto square =
            ormi::core::ops::mul(transposed, transposed);
        const auto added =
            ormi::core::ops::add(square, contiguous);
        const auto mapped =
            ormi::core::ops::mul(added, transposed);
        const auto output = ormi::core::ops::sum(mapped);
        const auto result = *static_cast<const float*>(output->data());
        const auto expected =
            static_cast<float>(rows * columns) * 0.140625f;

        EXPECT_FLOAT_EQ(result, expected);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesTiledMixedFloat64MeanWithPartialEdges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 65;
        constexpr std::size_t columns = 97;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{columns, rows},
            ormi::core::DType::float64
        );
        auto contiguous = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{rows, columns},
            ormi::core::DType::float64
        );
        auto* base_values = static_cast<double*>(base->data());
        auto* contiguous_values =
            static_cast<double*>(contiguous->data());
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                base_values[column * rows + row] =
                    static_cast<double>(
                        static_cast<int>((row + column * 3) % 13) - 6
                    ) * 0.03125;
                contiguous_values[row * columns + column] =
                    static_cast<double>(
                        static_cast<int>((row * 2 + column) % 11) - 5
                    ) * 0.015625;
            }
        }

        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto square =
            ormi::core::ops::mul(transposed, transposed);
        const auto added =
            ormi::core::ops::add(square, contiguous);
        const auto mapped =
            ormi::core::ops::mul(added, transposed);
        const auto output = ormi::core::ops::mean(mapped);
        const auto result = *static_cast<const double*>(output->data());

        double expected = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const auto lhs = base_values[column * rows + row];
                const auto rhs =
                    contiguous_values[row * columns + column];
                expected += (lhs * lhs + rhs) * lhs;
            }
        }
        expected /= static_cast<double>(rows * columns);

        EXPECT_NEAR(result, expected, 1.0e-10);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16ScalarMapSumWithTail) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto scale = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{
                static_cast<float>(static_cast<int>(i % 11) - 5) * 0.03125f
            };
        }
        const auto factor = Float16{0.5f};
        *static_cast<Float16*>(scale->data()) = factor;

        const auto scaled = ormi::core::ops::mul(input, scale);
        const auto mapped = ormi::core::ops::add(scaled, scale);
        const auto output = ormi::core::ops::sum(mapped);
        const auto result = *static_cast<const Float16*>(output->data());

        float expected = 0.0f;
        for (const auto value : std::span<const Float16>(values, input->numel())) {
            expected += static_cast<float>(value * factor + factor);
        }
        EXPECT_EQ(result.bits(), Float16{expected}.bits());
        EXPECT_FALSE(scaled->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesFloat16ReductionWithBaselineTarget) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "baseline");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{static_cast<float>(i % 8) * 0.03125f};
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped);
        const auto result = *static_cast<const Float16*>(output->data());

        float expected = 0.0f;
        for (const auto value : std::span<const Float16>(values, input->numel())) {
            expected += static_cast<float>(value * value + value);
        }
        EXPECT_EQ(result.bits(), Float16{expected}.bits());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesPointwiseChainIntoGlobalSum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        const auto input = tensor<float>(ormi::core::DType::float32);
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto mapped = ormi::core::ops::sin(added);
        const auto output = ormi::core::ops::sum(mapped);

        const auto value = *static_cast<const float*>(output->data());
        const float input_values[]{-0.5f, 0.0f, 0.5f, 1.0f};
        float expected = 0.0f;
        for (const auto input_value : input_values) {
            expected += std::sin(input_value * input_value + input_value);
        }

        EXPECT_FLOAT_EQ(value, expected);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
        EXPECT_FALSE(mapped->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST(CPUJITKernelTest, preservesFloat16MapSumRounding) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        values[0] = Float16{-0.5f};
        values[1] = Float16{0.1f};
        values[2] = Float16{0.5f};
        values[3] = Float16{1.0f};
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto mapped = ormi::core::ops::sin(added);
        const auto output = ormi::core::ops::sum(mapped);

        float expected = 0.0f;
        for (const auto value : std::span<const Float16>(values, 4)) {
            const auto rounded_square = value * value;
            const auto rounded_sum = rounded_square + value;
            expected += static_cast<float>(Float16{std::sin(static_cast<float>(rounded_sum))});
        }

        const auto result = *static_cast<const Float16*>(output->data());
        EXPECT_EQ(result.bits(), Float16{expected}.bits());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesPointwiseChainIntoPartialSum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        values[0] = 1.0f;
        values[1] = 2.0f;
        values[2] = 3.0f;
        values[3] = 4.0f;
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {0});

        const auto* result = static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(result[0], 14.0f);
        EXPECT_FLOAT_EQ(result[1], 26.0f);
        EXPECT_FALSE(mapped->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST(CPUJITKernelTest, fusesPointwiseChainIntoPartialMean) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        values[0] = 1.0f;
        values[1] = 2.0f;
        values[2] = 3.0f;
        values[3] = 4.0f;
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped, {0});

        const auto* result = static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(result[0], 7.0f);
        EXPECT_FLOAT_EQ(result[1], 13.0f);
        EXPECT_FALSE(mapped->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST(CPUJITKernelTest, executesVectorizedArithmeticGlobalSumWithTail) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        float expected = 0.0f;
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(i + 1) * 0.125f;
            expected += values[i] * values[i] + values[i];
        }
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped);

        const auto result = *static_cast<const float*>(output->data());

        EXPECT_NEAR(result, expected, 1.0e-5f);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesVectorizedLastAxisFloat64MeanWithTail) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 33},
            ormi::core::DType::float64
        );
        auto* values = static_cast<double*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<double>(i + 1) * 0.25;
        }
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped, {1});

        const auto* result = static_cast<const double*>(output->data());
        for (std::size_t row = 0; row < 2; ++row) {
            double expected = 0.0;
            for (std::size_t column = 0; column < 33; ++column) {
                const auto value = values[row * 33 + column];
                expected += value * value + value;
            }
            EXPECT_NEAR(result[row], expected / 33.0, 1.0e-12);
        }
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesLargeLastAxisFloat32SumAcrossRanges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 1024;
        constexpr std::size_t columns = 513;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{rows, columns},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                values[row * columns + column] =
                    static_cast<float>(
                        static_cast<int>((row + column) % 17) - 8
                    ) * 0.03125f;
            }
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t row = 0; row < rows; ++row) {
            float expected = 0.0f;
            for (std::size_t column = 0; column < columns; ++column) {
                const auto value = values[row * columns + column];
                expected += value * value + value;
            }
            EXPECT_NEAR(result[row], expected, 1.0e-5f);
        }
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesLargeLastAxisFloat16SumAcrossRanges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        using ormi::core::Float16;
        constexpr std::size_t rows = 1024;
        constexpr std::size_t columns = 513;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{rows, columns},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{0.5f};
        }

        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto* result = static_cast<const Float16*>(output->data());

        const Float16 expected{384.75f};
        for (std::size_t row = 0; row < rows; ++row) {
            EXPECT_EQ(result[row].bits(), expected.bits());
        }
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesFloat16PartialMean) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        values[0] = Float16{0.5f};
        values[1] = Float16{1.0f};
        values[2] = Float16{1.5f};
        values[3] = Float16{2.0f};
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto* result = static_cast<const Float16*>(output->data());

        EXPECT_EQ(result[0].bits(), Float16{1.375f}.bits());
        EXPECT_EQ(result[1].bits(), Float16{4.875f}.bits());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesFloat16PartialSum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 2},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        values[0] = Float16{0.5f};
        values[1] = Float16{1.0f};
        values[2] = Float16{1.5f};
        values[3] = Float16{2.0f};
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto* result = static_cast<const Float16*>(output->data());

        EXPECT_EQ(result[0].bits(), Float16{2.75f}.bits());
        EXPECT_EQ(result[1].bits(), Float16{9.75f}.bits());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesAutocastIntoFloat32Sum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        values[0] = Float16{0.5f};
        values[1] = Float16{1.0f};

        std::shared_ptr<ormi::core::Tensor> mapped;
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            mapped = ormi::core::ops::sin(input);
            output = ormi::core::ops::sum(mapped);
        }

        const auto result = *static_cast<const float*>(output->data());
        const auto expected = std::sin(0.5f) + std::sin(1.0f);

        EXPECT_FLOAT_EQ(result, expected);
        EXPECT_EQ(output->dtype(), ormi::core::DType::float32);
        EXPECT_FALSE(mapped->realized());
        EXPECT_FALSE(mapped->parents()[0]->realized());
    }

    TEST(CPUJITKernelTest, executesPackedAutocastFloat16ArithmeticSum) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{
                static_cast<float>(static_cast<int>(i % 17) - 8) * 0.03125f
            };
        }

        std::shared_ptr<ormi::core::Tensor> mapped;
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            const auto square = ormi::core::ops::mul(input, input);
            mapped = ormi::core::ops::add(square, input);
            output = ormi::core::ops::sum(mapped);
        }

        const auto result = *static_cast<const float*>(output->data());
        float expected = 0.0f;
        for (const auto value : std::span<const Float16>(values, input->numel())) {
            expected += static_cast<float>(value * value + value);
        }

        EXPECT_NEAR(result, expected, 1.0e-5f);
        EXPECT_EQ(output->dtype(), ormi::core::DType::float32);
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat32Float16RoundTrip) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(i) * 0.013f - 0.4f;
        }

        const auto narrowed = ormi::core::ops::cast(
            input,
            ormi::core::DType::float16
        );
        const auto square = ormi::core::ops::mul(narrowed, narrowed);
        const auto output = ormi::core::ops::cast(
            square,
            ormi::core::DType::float32
        );
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const Float16 value{values[i]};
            EXPECT_FLOAT_EQ(result[i], static_cast<float>(value * value));
        }
        EXPECT_FALSE(narrowed->realized());
        EXPECT_FALSE(square->realized());
    }

    TEST(CPUJITKernelTest, roundsExplicitFloat16CastInsideFusion) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        values[0] = 0.3333f;
        values[1] = 0.6667f;
        const auto narrowed = ormi::core::ops::cast(input, ormi::core::DType::float16);
        const auto output = ormi::core::ops::sin(narrowed);
        const auto* result = static_cast<const ormi::core::Float16*>(output->data());

        EXPECT_EQ(
            result[0].bits(),
            ormi::core::Float16{std::sin(static_cast<float>(ormi::core::Float16{values[0]}))}.bits()
        );
        EXPECT_EQ(
            result[1].bits(),
            ormi::core::Float16{std::sin(static_cast<float>(ormi::core::Float16{values[1]}))}.bits()
        );
        EXPECT_FALSE(narrowed->realized());
    }

    TEST(CPUJITKernelTest, fusesFloat64ToFloat32Cast) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float64
        );
        auto* values = static_cast<double*>(input->data());
        values[0] = 0.333333333333;
        values[1] = 0.666666666667;
        const auto narrowed = ormi::core::ops::cast(input, ormi::core::DType::float32);
        const auto output = ormi::core::ops::sin(narrowed);
        const auto* result = static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(result[0], std::sin(static_cast<float>(values[0])));
        EXPECT_FLOAT_EQ(result[1], std::sin(static_cast<float>(values[1])));
        EXPECT_FALSE(narrowed->realized());
    }

    TEST(CPUJITKernelTest, fusesMultiAxisSumWithKeepdims) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3, 2},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(i + 1);
        }
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {0, 2}, true);
        const auto* result = static_cast<const float*>(output->data());

        EXPECT_EQ(output->shape(), (ormi::core::Shape{1, 3, 1}));
        EXPECT_FLOAT_EQ(result[0], 136.0f);
        EXPECT_FLOAT_EQ(result[1], 232.0f);
        EXPECT_FLOAT_EQ(result[2], 360.0f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesSeparatedAxesMeanWithTail) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t first = 3;
        constexpr std::size_t middle = 4;
        constexpr std::size_t last = 65;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{first, middle, last},
            ormi::core::DType::float32
        );
        auto row = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{last},
            ormi::core::DType::float32
        );
        auto column = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{middle, 1},
            ormi::core::DType::float32
        );
        auto* input_values = static_cast<float*>(input->data());
        auto* row_values = static_cast<float*>(row->data());
        auto* column_values = static_cast<float*>(column->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            input_values[i] = static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.03125f;
        }
        for (std::size_t i = 0; i < last; ++i) {
            row_values[i] = static_cast<float>(i % 5 + 1) * 0.125f;
        }
        for (std::size_t i = 0; i < middle; ++i) {
            column_values[i] = static_cast<float>(
                static_cast<int>(i) - 2
            ) * 0.0625f;
        }

        const auto shifted = ormi::core::ops::add(
            ormi::core::ops::mul(input, row),
            column
        );
        const auto mapped = ormi::core::ops::mul(shifted, shifted);
        const auto output = ormi::core::ops::mean(mapped, {0, 2});
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t j = 0; j < middle; ++j) {
            double expected = 0.0;
            for (std::size_t i = 0; i < first; ++i) {
                for (std::size_t k = 0; k < last; ++k) {
                    const auto input_index =
                        (i * middle + j) * last + k;
                    const auto shifted_value =
                        input_values[input_index] * row_values[k] +
                        column_values[j];
                    expected += shifted_value * shifted_value;
                }
            }
            expected /= static_cast<double>(first * last);
            EXPECT_NEAR(result[j], static_cast<float>(expected), 1.0e-6f);
        }
        EXPECT_FALSE(shifted->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesTransposedBroadcastPartialSum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3},
            ormi::core::DType::float32
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        auto* base_values = static_cast<float*>(base->data());
        for (std::size_t i = 0; i < base->numel(); ++i) {
            base_values[i] = static_cast<float>(i + 1);
        }
        auto* rhs_values = static_cast<float*>(rhs->data());
        rhs_values[0] = 0.5f;
        rhs_values[1] = 1.0f;

        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto square = ormi::core::ops::mul(transposed, transposed);
        const auto mapped = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto* result = static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(result[0], 18.5f);
        EXPECT_FLOAT_EQ(result[1], 30.5f);
        EXPECT_FLOAT_EQ(result[2], 46.5f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesLargeTransposedSumAcrossRanges) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 1024;
        constexpr std::size_t columns = 513;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{columns, rows},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(base->data());
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t row = 0; row < rows; ++row) {
                values[column * rows + row] =
                    static_cast<float>(
                        static_cast<int>((row + column) % 17) - 8
                    ) * 0.03125f;
            }
        }

        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto square = ormi::core::ops::mul(transposed, transposed);
        const auto mapped = ormi::core::ops::add(square, transposed);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t row = 0; row < rows; ++row) {
            float expected = 0.0f;
            for (std::size_t column = 0; column < columns; ++column) {
                const auto value = values[column * rows + row];
                expected += value * value + value;
            }
            EXPECT_NEAR(result[row], expected, 1.0e-5f);
        }
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesTiledTransposedFloat64MeanWithTail) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t rows = 65;
        constexpr std::size_t columns = 257;
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{columns, rows},
            ormi::core::DType::float64
        );
        auto* values = static_cast<double*>(base->data());
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t row = 0; row < rows; ++row) {
                values[column * rows + row] =
                    static_cast<double>(
                        static_cast<int>((row + column) % 9) - 4
                    ) * 0.0625;
            }
        }

        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto square = ormi::core::ops::mul(transposed, transposed);
        const auto mapped = ormi::core::ops::add(square, transposed);
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto* result = static_cast<const double*>(output->data());

        for (std::size_t row = 0; row < rows; ++row) {
            double expected = 0.0;
            for (std::size_t column = 0; column < columns; ++column) {
                const auto value = values[column * rows + row];
                expected += value * value + value;
            }
            expected /= static_cast<double>(columns);
            EXPECT_NEAR(result[row], expected, 1.0e-12);
        }
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesPermutedMultiAxisSum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3, 4},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(base->data());
        for (std::size_t i = 0; i < base->numel(); ++i) {
            values[i] = static_cast<float>(i + 1) * 0.1f;
        }

        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(permuted, permuted),
            permuted
        );
        const auto output = ormi::core::ops::sum(mapped, {0, 2});
        const auto* result = static_cast<const float*>(output->data());

        for (std::size_t axis1 = 0; axis1 < 2; ++axis1) {
            float expected = 0.0f;
            for (std::size_t axis0 = 0; axis0 < 4; ++axis0) {
                for (std::size_t axis2 = 0; axis2 < 3; ++axis2) {
                    const auto input_index =
                        axis1 * 12 + axis2 * 4 + axis0;
                    const auto value = values[input_index];
                    expected += value * value + value;
                }
            }
            EXPECT_FLOAT_EQ(result[axis1], expected);
        }
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, fusesExpandedPartialSum) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 1},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(base->data());
        values[0] = 0.25f;
        values[1] = 0.5f;

        const auto expanded = ormi::core::ops::expand(base, {2, 3});
        const auto added = ormi::core::ops::add(expanded, expanded);
        const auto mapped = ormi::core::ops::sin(added);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto* result = static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(result[0], std::sin(0.5f) * 3.0f);
        EXPECT_FLOAT_EQ(result[1], std::sin(1.0f) * 3.0f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, supportsMoreThanEightInputs) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        std::vector<std::shared_ptr<ormi::core::Tensor>> inputs;
        inputs.reserve(9);

        for (std::size_t input = 0; input < 9; ++input) {
            auto tensor = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(),
                ormi::core::Shape{2},
                ormi::core::DType::float32
            );
            auto* values = static_cast<float*>(tensor->data());
            values[0] = static_cast<float>(input + 1);
            values[1] = static_cast<float>((input + 1) * 2);
            inputs.push_back(std::move(tensor));
        }

        auto output = inputs[0];
        for (std::size_t input = 1; input < inputs.size(); ++input) {
            output = ormi::core::ops::add(output, inputs[input]);
        }
        const auto* result = static_cast<const float*>(output->data());

        EXPECT_FLOAT_EQ(result[0], 45.0f);
        EXPECT_FLOAT_EQ(result[1], 90.0f);
    }

    TEST(CPUJITKernelTest, publishesCompiledKernelAcrossThreads) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "");
        constexpr std::size_t workers = 4;
        std::array<std::shared_ptr<ormi::core::Tensor>, workers> outputs;
        std::array<std::exception_ptr, workers> errors{};

        for (std::size_t worker = 0; worker < workers; ++worker) {
            auto input = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(),
                ormi::core::Shape{2},
                ormi::core::DType::float32
            );
            auto* values = static_cast<float*>(input->data());
            values[0] = static_cast<float>(worker + 1);
            values[1] = static_cast<float>(worker + 2);
            const auto square = ormi::core::ops::mul(input, input);
            outputs[worker] = ormi::core::ops::add(square, input);
        }

        std::array<std::thread, workers> threads;
        for (std::size_t worker = 0; worker < workers; ++worker) {
            threads[worker] = std::thread([&, worker] {
                try {
                    outputs[worker]->realize();
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        for (std::size_t worker = 0; worker < workers; ++worker) {
            if (errors[worker]) {
                std::rethrow_exception(errors[worker]);
            }
            const auto* result =
                static_cast<const float*>(outputs[worker]->data());
            const auto first = static_cast<float>(worker + 1);
            const auto second = static_cast<float>(worker + 2);
            EXPECT_FLOAT_EQ(result[0], first * first + first);
            EXPECT_FLOAT_EQ(result[1], second * second + second);
        }
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

#if defined(_WIN32)
    TEST(CPUJITKernelTest, executesBroadcastSpanWithConfiguredMSVC) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "msvc");
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 8},
            ormi::core::DType::float32
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{8},
            ormi::core::DType::float32
        );
        auto* lhs_values = static_cast<float*>(lhs->data());
        auto* rhs_values = static_cast<float*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = static_cast<float>(i + 1) * 0.25f;
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            rhs_values[i] = static_cast<float>(i + 1) * 0.5f;
        }

        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const float* result = nullptr;

        try {
            result = static_cast<const float*>(output->data());
        } catch (const ormi::jitError& error) {
            GTEST_SKIP() << "Configured MSVC is unavailable: " << error.what();
        }

        for (std::size_t i = 0; i < output->numel(); ++i) {
            EXPECT_FLOAT_EQ(
                result[i],
                (lhs_values[i] * lhs_values[i] + rhs_values[i % 8]) * lhs_values[i]
            );
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesVectorizedReductionWithConfiguredMSVC) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "msvc");
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float32
        );
        auto* values = static_cast<float*>(input->data());
        float expected = 0.0f;
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = static_cast<float>(i + 1) * 0.125f;
            expected += values[i] * values[i] + values[i];
        }
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped);
        const float* result = nullptr;

        try {
            result = static_cast<const float*>(output->data());
        } catch (const ormi::jitError& error) {
            GTEST_SKIP() << "Configured MSVC is unavailable: " << error.what();
        }

        EXPECT_NEAR(*result, expected, 1.0e-5f);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16WithConfiguredMSVC) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "msvc");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{static_cast<float>(i) * 0.03125f};
        }
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mul(added, input);
        const Float16* result = nullptr;

        try {
            result = static_cast<const Float16*>(output->data());
        } catch (const ormi::jitError& error) {
            GTEST_SKIP() << "Configured MSVC is unavailable: " << error.what();
        }

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected = (values[i] * values[i] + values[i]) * values[i];
            EXPECT_EQ(result[i].bits(), expected.bits());
        }
    }

    TEST(CPUJITKernelTest, executesPackedFloat16ReductionWithConfiguredMSVC) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "msvc");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{65},
            ormi::core::DType::float16
        );
        auto* values = static_cast<Float16*>(input->data());
        for (std::size_t i = 0; i < input->numel(); ++i) {
            values[i] = Float16{
                static_cast<float>(static_cast<int>(i % 17) - 8) * 0.03125f
            };
        }
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto mapped = ormi::core::ops::mul(added, input);
        const auto output = ormi::core::ops::sum(mapped);
        const Float16* result = nullptr;

        try {
            result = static_cast<const Float16*>(output->data());
        } catch (const ormi::jitError& error) {
            GTEST_SKIP() << "Configured MSVC is unavailable: " << error.what();
        }

        float expected = 0.0f;
        for (const auto value : std::span<const Float16>(values, input->numel())) {
            expected += static_cast<float>((value * value + value) * value);
        }
        EXPECT_EQ(result->bits(), Float16{expected}.bits());
        EXPECT_FALSE(mapped->realized());
    }

    TEST(CPUJITKernelTest, executesPackedFloat16BroadcastWithConfiguredMSVC) {
        if (!ormi::platform::supports_f16c()) {
            GTEST_SKIP() << "F16C is unavailable on this CPU.";
        }

        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "msvc");
        environment("ORMI_CPU_JIT_TARGET", "auto");
        using ormi::core::Float16;
        auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4, 17},
            ormi::core::DType::float16
        );
        auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{17},
            ormi::core::DType::float16
        );
        auto* lhs_values = static_cast<Float16*>(lhs->data());
        auto* rhs_values = static_cast<Float16*>(rhs->data());
        for (std::size_t i = 0; i < lhs->numel(); ++i) {
            lhs_values[i] = Float16{static_cast<float>(i % 17) * 0.03125f};
        }
        for (std::size_t i = 0; i < rhs->numel(); ++i) {
            rhs_values[i] = Float16{static_cast<float>(i + 1) * 0.015625f};
        }
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const Float16* result = nullptr;

        try {
            result = static_cast<const Float16*>(output->data());
        } catch (const ormi::jitError& error) {
            GTEST_SKIP() << "Configured MSVC is unavailable: " << error.what();
        }

        for (std::size_t i = 0; i < output->numel(); ++i) {
            const auto expected =
                (lhs_values[i] * lhs_values[i] + rhs_values[i % 17]) * lhs_values[i];
            EXPECT_EQ(result[i].bits(), expected.bits());
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
    }

    TEST(CPUJITKernelTest, executesWithConfiguredMSVCOutsideDeveloperShell) {
        Cache cache;
        environment("ORMI_CPU_JIT_CXX", "");
        environment("ORMI_CPU_JIT_CXX_ID", "msvc");

        const auto input = tensor<float>(ormi::core::DType::float32);
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto product = ormi::core::ops::mul(added, input);
        const auto output = ormi::core::ops::neg(product);
        const float* data = nullptr;

        try {
            data = static_cast<const float*>(output->data());
        } catch (const ormi::jitError& error) {
            GTEST_SKIP() << "Configured MSVC is unavailable: " << error.what();
        }

        EXPECT_FLOAT_EQ(data[0], -0.125f);
        EXPECT_FLOAT_EQ(data[1], 0.0f);
        EXPECT_FLOAT_EQ(data[2], -0.375f);
        EXPECT_FLOAT_EQ(data[3], -2.0f);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
        EXPECT_FALSE(product->realized());
    }
#endif
}
