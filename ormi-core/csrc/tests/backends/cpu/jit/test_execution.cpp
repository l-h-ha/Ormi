#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/executor.h"
#include "backends/cpu/jit/compile.h"
#include "backends/cpu/jit/execute.h"

namespace {
    namespace cpu = ormi::backends::cpu;
    namespace fusion = ormi::core::runtime::fusion;
    using ormi::core::DType;
    using ormi::core::Device;
    using ormi::core::Shape;
    using ormi::core::Tensor;

    struct Outputs {
        std::size_t inputs;
        std::size_t outputs;
    };

    void map(void* const* outputs, const void* const* inputs,
        unsigned long long, unsigned long long begin, unsigned long long end,
        const void* math) {
        const auto& counts = *static_cast<const Outputs*>(math);
        for (auto i = begin; i < end; ++i) {
            float value = 0;
            for (std::size_t input = 0; input < counts.inputs; ++input) {
                value += static_cast<const float*>(inputs[input])[i];
            }
            for (std::size_t output = 0; output < counts.outputs; ++output) {
                static_cast<float*>(outputs[output])[i] = value + static_cast<float>(output);
            }
        }
    }

    void output_range(void* const* outputs, const void* const*,
        unsigned long long elements, unsigned long long begin, unsigned long long end,
        const void*) {
        for (auto i = begin; i < end; ++i) {
            static_cast<float*>(outputs[0])[i] = static_cast<float>(elements + i);
        }
    }

    template <typename T>
    void partial(void* const* outputs, const void* const* inputs,
        unsigned long long, unsigned long long begin, unsigned long long end,
        const void*) {
        auto* value = static_cast<T*>(outputs[0]);
        const auto* input = static_cast<const T*>(inputs[0]);
        for (auto i = begin; i < end; ++i) {
            *value += input[i];
        }
    }

    fusion::Expr expression(DType dtype, std::size_t elements, std::vector<Tensor*> inputs) {
        return fusion::Expr{
            .dtype = dtype,
            .shape = {elements},
            .nodes = {},
            .inputs = std::move(inputs),
            .output = 0,
            .auxiliary_outputs = {},
            .operations = 0,
            .elements = elements,
            .reduction = std::nullopt
        };
    }

    TEST(CPUJITExecutionTest, refreshesPointersBeyondStackCapacities) {
        constexpr std::size_t elements = 65;
        std::vector<std::unique_ptr<Tensor>> inputs;
        std::vector<std::unique_ptr<Tensor>> outputs;
        std::vector<Tensor*> input_pointers;
        std::vector<Tensor*> auxiliary;
        for (std::size_t i = 0; i < 9; ++i) {
            inputs.push_back(std::make_unique<Tensor>(Device::cpu(), Shape{elements}, DType::float32));
            input_pointers.push_back(inputs.back().get());
            std::fill_n(static_cast<float*>(inputs.back()->data()), elements, static_cast<float>(i + 1));
        }
        for (std::size_t i = 0; i < 5; ++i) {
            outputs.push_back(std::make_unique<Tensor>(Device::cpu(), Shape{elements}, DType::float32));
            if (i != 0) {
                auxiliary.push_back(outputs.back().get());
            }
        }
        const Outputs counts{inputs.size(), outputs.size()};
        const auto expr = expression(DType::float32, elements, input_pointers);
        auto compiled = std::make_shared<cpu::jit::Compiled>(cpu::jit::Compiled{
            nullptr, map, &counts, elements, 8, 0, false
        });
        auto setup = cpu::jit::execution::prepare(auxiliary, expr, compiled);
        std::weak_ptr<cpu::jit::Compiled> retained = compiled;

        cpu::jit::execution::run(*outputs[0], auxiliary, expr, *compiled);
        for (std::size_t output = 0; output < outputs.size(); ++output) {
            for (const auto value : std::span(static_cast<float*>(outputs[output]->data()), elements)) {
                EXPECT_FLOAT_EQ(value, 45.0f + static_cast<float>(output));
            }
        }

        *inputs[0] = Tensor(Device::cpu(), Shape{elements}, DType::float32);
        std::fill_n(static_cast<float*>(inputs[0]->data()), elements, 11.0f);
        for (auto& output : outputs) {
            *output = Tensor(Device::cpu(), Shape{elements}, DType::float32);
        }
        compiled.reset();
        auto moved = std::move(setup);
        moved.run(*outputs[0]);
        for (std::size_t output = 0; output < outputs.size(); ++output) {
            for (const auto value : std::span(static_cast<float*>(outputs[output]->data()), elements)) {
                EXPECT_FLOAT_EQ(value, 55.0f + static_cast<float>(output));
            }
        }
        EXPECT_FALSE(retained.expired());
        moved = {};
        EXPECT_TRUE(retained.expired());
    }

    TEST(CPUJITExecutionTest, usesReductionOutputRangeForUnpartitionedKernels) {
        Tensor output(Device::cpu(), Shape{7}, DType::float32);
        auto expr = expression(DType::float32, 35, {});
        expr.reduction = fusion::Reduction{
            ormi::core::ops::ReduceOp::sum, {7, {}}, {5, {}}
        };
        auto compiled = std::make_shared<cpu::jit::Compiled>(cpu::jit::Compiled{
            nullptr, output_range, nullptr, 35, 0, 9999, false
        });
        auto setup = cpu::jit::execution::prepare({}, expr, compiled);
        for (const bool prepared : {false, true}) {
            std::fill_n(static_cast<float*>(output.data()), 7, -1.0f);
            if (prepared) {
                setup.run(output);
            } else {
                cpu::jit::execution::run(output, {}, expr, *compiled);
            }
            for (std::size_t i = 0; i < 7; ++i) {
                EXPECT_FLOAT_EQ(static_cast<float*>(output.data())[i], 35.0f + static_cast<float>(i));
            }
        }
    }

    TEST(CPUJITExecutionTest, retainsBoundsAfterExpressionChanges) {
        Tensor output(Device::cpu(), Shape{7}, DType::float32);
        auto expr = expression(DType::float32, 7, {});
        auto compiled = std::make_shared<cpu::jit::Compiled>(cpu::jit::Compiled{
            nullptr, output_range, nullptr, 7, 1, 0, false
        });
        auto setup = cpu::jit::execution::prepare({}, expr, compiled);
        expr.elements = 9999;
        expr.inputs.clear();
        setup.run(output);
        for (std::size_t i = 0; i < 7; ++i) {
            EXPECT_FLOAT_EQ(static_cast<float*>(output.data())[i], 7.0f + static_cast<float>(i));
        }
    }

    TEST(CPUJITExecutionTest, partitionsMapRangesIncludingFinalTail) {
        constexpr std::size_t elements = 262147;
        Tensor input(Device::cpu(), Shape{elements}, DType::float32);
        Tensor output(Device::cpu(), Shape{elements}, DType::float32);
        std::fill_n(static_cast<float*>(input.data()), elements, 2.0f);
        const Outputs counts{1, 1};
        const auto expr = expression(DType::float32, elements, {&input});
        auto compiled = std::make_shared<cpu::jit::Compiled>(cpu::jit::Compiled{
            nullptr, map, &counts, elements * 4, 16, 0, false
        });
        auto setup = cpu::jit::execution::prepare({}, expr, compiled);
        for (const bool prepared : {false, true}) {
            std::fill_n(static_cast<float*>(output.data()), elements, -1.0f);
            if (prepared) {
                setup.run(output);
            } else {
                cpu::jit::execution::run(output, {}, expr, *compiled);
            }
            for (std::size_t i = 0; i < elements; ++i) {
                if (static_cast<float*>(output.data())[i] != 2.0f) {
                    FAIL() << "Unwritten or incorrect map element " << i;
                }
            }
        }
    }

    template <typename T>
    void check_partial(DType dtype, bool parallel) {
        constexpr std::size_t range = 17;
        Tensor input(Device::cpu(), Shape{range}, std::is_same_v<T, double> ? DType::float64 : DType::float32);
        Tensor output(Device::cpu(), Shape{}, dtype);
        std::fill_n(static_cast<T*>(input.data()), range, T{1});
        auto expr = expression(dtype, range * 3, {&input});
        expr.reduction = fusion::Reduction{
            ormi::core::ops::ReduceOp::mean, {1, {}}, {range, {}}
        };
        auto compiled = std::make_shared<cpu::jit::Compiled>(cpu::jit::Compiled{
            nullptr, partial<T>, nullptr, parallel ? 1048576u : 17u, 4, range, true
        });
        const auto slots = cpu::executor::slots(range, compiled->unit, compiled->work);
        for (const auto op : {ormi::core::ops::ReduceOp::sum, ormi::core::ops::ReduceOp::mean}) {
            expr.reduction->op = op;
            auto setup = cpu::jit::execution::prepare({}, expr, compiled);
            EXPECT_EQ(setup.buffers(), slots > 1 ? 1u : 0u);
            EXPECT_EQ(setup.bytes(), slots > 1 ? slots * sizeof(T) : 0u);
            for (const T value : {T{1.00048828125}, T{2}, T{-1}}) {
                std::fill_n(static_cast<T*>(input.data()), range, value);
                const auto expected = op == ormi::core::ops::ReduceOp::sum ? value * range : value;
                for (const bool prepared : {false, true}) {
                    if (prepared) {
                        setup.run(output);
                    } else {
                        cpu::jit::execution::run(output, {}, expr, *compiled);
                    }
                    if (dtype == DType::float16) {
                        EXPECT_EQ(static_cast<ormi::core::Float16*>(output.data())->bits(),
                            ormi::core::Float16(static_cast<float>(expected)).bits());
                    } else {
                        EXPECT_EQ(*static_cast<T*>(output.data()), expected);
                    }
                }
            }
        }
    }

    TEST(CPUJITExecutionTest, preservesEmptySumAndMean) {
        Tensor input(Device::cpu(), Shape{0}, DType::float32);
        Tensor output(Device::cpu(), Shape{}, DType::float32);
        auto expr = expression(DType::float32, 0, {&input});
        expr.reduction = fusion::Reduction{
            ormi::core::ops::ReduceOp::sum, {1, {}}, {0, {}}
        };
        auto compiled = std::make_shared<cpu::jit::Compiled>(cpu::jit::Compiled{
            nullptr, partial<float>, nullptr, 0, 1, 0, true
        });
        for (const auto op : {ormi::core::ops::ReduceOp::sum, ormi::core::ops::ReduceOp::mean}) {
            expr.reduction->op = op;
            auto setup = cpu::jit::execution::prepare({}, expr, compiled);
            for (const bool prepared : {false, true}) {
                if (prepared) {
                    setup.run(output);
                } else {
                    cpu::jit::execution::run(output, {}, expr, *compiled);
                }
                const auto value = *static_cast<float*>(output.data());
                if (op == ormi::core::ops::ReduceOp::sum) {
                    EXPECT_FLOAT_EQ(value, 0.0f);
                } else {
                    EXPECT_TRUE(std::isnan(value));
                }
            }
        }
    }

    TEST(CPUJITExecutionTest, combinesAndNormalizesFloat32Partials) {
        check_partial<float>(DType::float32, false);
        check_partial<float>(DType::float32, true);
    }

    TEST(CPUJITExecutionTest, combinesAndNormalizesFloat64Partials) {
        check_partial<double>(DType::float64, false);
        check_partial<double>(DType::float64, true);
    }

    TEST(CPUJITExecutionTest, roundsFloat16AfterCombiningFloat32Partials) {
        check_partial<float>(DType::float16, false);
        check_partial<float>(DType::float16, true);
    }

    TEST(CPUJITExecutionTest, rejectsAuxiliaryOutputsForPartialReduction) {
        Tensor output(Device::cpu(), Shape{}, DType::float32);
        Tensor auxiliary(Device::cpu(), Shape{}, DType::float32);
        auto expr = expression(DType::float32, 17, {});
        expr.reduction = fusion::Reduction{
            ormi::core::ops::ReduceOp::sum, {1, {}}, {17, {}}
        };
        auto compiled = std::make_shared<cpu::jit::Compiled>(cpu::jit::Compiled{
            nullptr, partial<float>, nullptr, 17, 4, 17, true
        });
        std::array<Tensor*, 1> outputs{&auxiliary};
        auto setup = cpu::jit::execution::prepare(outputs, expr, compiled);
        EXPECT_THROW(cpu::jit::execution::run(output, outputs, expr, *compiled), ormi::OperationError);
        EXPECT_THROW(setup.run(output), ormi::OperationError);
    }

    TEST(CPUJITExecutionTest, rejectsNonFloatingPartialReduction) {
        Tensor output(Device::cpu(), Shape{}, DType::int32);
        const auto expr = expression(DType::int32, 17, {});
        auto compiled = std::make_shared<cpu::jit::Compiled>(cpu::jit::Compiled{
            nullptr, partial<float>, nullptr, 17, 4, 17, true
        });
        auto setup = cpu::jit::execution::prepare({}, expr, compiled);
        EXPECT_THROW(cpu::jit::execution::run(output, {}, expr, *compiled), ormi::DTypeError);
        EXPECT_THROW(setup.run(output), ormi::DTypeError);
    }
}
