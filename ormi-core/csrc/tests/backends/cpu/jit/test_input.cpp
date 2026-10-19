#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ormi/core/error.h>

#include "backends/cpu/jit/codegen/input.h"

namespace {
    namespace core = ormi::core;
    namespace fusion = core::runtime::fusion;
    namespace jit = ormi::backends::cpu::jit;
    namespace codegen = jit::codegen;

    TEST(CPUJITInputTest, preservesOffsetBytesAndAxisOrder) {
        EXPECT_EQ(codegen::offset("i", {}), "0");
        EXPECT_EQ(codegen::offset("i", {{1, 65, 1}}), "(i % 65)");
        EXPECT_EQ(
            codegen::offset("base", {{195, 5, 390}, {65, 3, 1}, {1, 65, 2}}),
            "((base / 195) % 5) * 390 + ((base / 65) % 3) + (base % 65) * 2"
        );
    }

    TEST(CPUJITInputTest, retainsRankIndependentOffsetFormatting) {
        std::vector<fusion::Axis> axes;
        std::string expected;
        for (std::size_t dimension = 0; dimension < 16; ++dimension) {
            axes.push_back({dimension + 2, dimension + 3, dimension + 4});
            if (dimension != 0) expected += " + ";
            expected += "((output_index / " + std::to_string(dimension + 2) +
                ") % " + std::to_string(dimension + 3) + ") * " +
                std::to_string(dimension + 4);
        }
        EXPECT_EQ(codegen::offset("output_index", axes), expected);
    }

    TEST(CPUJITInputTest, preservesEveryScalarAccessMode) {
        const fusion::Input input{2, {{1, 65, 2}}, false};
        constexpr std::array modes{
            jit::Access::indexed, jit::Access::linear, jit::Access::scalar,
            jit::Access::span, jit::Access::fixed
        };
        const std::array<std::string, 5> expected{
            "input2[(i % 65) * 2]", "input2[i]", "input2_scalar",
            "input2_span[j]", "input2_fixed"
        };
        for (const auto dtype : {core::DType::float16, core::DType::float32, core::DType::float64}) {
            for (std::size_t i = 0; i < modes.size(); ++i) {
                SCOPED_TRACE(i);
                SCOPED_TRACE(static_cast<int>(dtype));
                jit::Loop loop;
                loop.vectorized = true;
                loop.inputs = {jit::Access::indexed, jit::Access::indexed, modes[i]};
                EXPECT_EQ(codegen::input_value(dtype, input, loop),
                    dtype == core::DType::float16 ? "ormi_half_value(" + expected[i] + ")" : expected[i]);
            }
        }
    }

    TEST(CPUJITInputTest, usesSpanRelativeLinearLoadsOnlyWhenPlanned) {
        const fusion::Input input{0, {}, true};
        jit::Loop loop;
        loop.vectorized = true;
        loop.inputs = {jit::Access::linear};
        loop.span = 65;
        EXPECT_EQ(codegen::input_value(core::DType::float32, input, loop), "input0_span[j]");
        EXPECT_EQ(codegen::packed_input(input, core::DType::float32, loop), "_mm256_loadu_ps(input0_span + j)");
        EXPECT_EQ(codegen::packed_input(input, core::DType::float16, loop), "ormi_half_load8(input0_span + j)");
        loop.vectorized = false;
        loop.inputs.clear();
        EXPECT_EQ(codegen::input_value(core::DType::float32, input, loop), "input0[i]");
    }

    TEST(CPUJITInputTest, scalarFallbackIgnoresPackedAccessAndKeepsIndexedAddress) {
        const fusion::Input input{0, {{1, 65, 2}}, false};
        jit::Loop loop;
        loop.span = 65;
        loop.inputs = {jit::Access::fixed};
        EXPECT_EQ(codegen::input_value(core::DType::float64, input, loop), "input0[(i % 65) * 2]");
        EXPECT_EQ(codegen::input_value(core::DType::float16, {0, {}, false}, loop), "ormi_half_value(input0[0])");
    }

    TEST(CPUJITInputTest, preservesEveryPackedAccessMode) {
        const fusion::Input input{2, {}, true};
        constexpr std::array modes{
            jit::Access::linear, jit::Access::scalar, jit::Access::span, jit::Access::fixed
        };
        const std::array<std::string, 4> single{
            "_mm256_loadu_ps(input2 + i)", "_mm256_set1_ps(input2_scalar)",
            "_mm256_loadu_ps(input2_span + j)", "_mm256_set1_ps(input2_fixed)"
        };
        const std::array<std::string, 4> half{
            "ormi_half_load8(input2 + i)", "_mm256_set1_ps(ormi_half_value(input2_scalar))",
            "ormi_half_load8(input2_span + j)", "_mm256_set1_ps(ormi_half_value(input2_fixed))"
        };
        for (std::size_t i = 0; i < modes.size(); ++i) {
            SCOPED_TRACE(i);
            jit::Loop loop;
            loop.inputs = {jit::Access::indexed, jit::Access::indexed, modes[i]};
            EXPECT_EQ(codegen::packed_input(input, core::DType::float32, loop), single[i]);
            EXPECT_EQ(codegen::packed_input(input, core::DType::float16, loop), half[i]);
        }
    }

    TEST(CPUJITInputTest, scalarStridedLoadsSkipBroadcastDimensions) {
        const fusion::Input input{1, {}, false};
        jit::Loop loop;
        loop.strided = true;
        loop.strides = {{}, {0, 1, 195, 0, 3}};
        const std::string expected = "input1[index1 + index2 * 195 + index4 * 3]";
        for (const auto dtype : {core::DType::float16, core::DType::float32, core::DType::float64}) {
            EXPECT_EQ(codegen::input_value(dtype, input, loop),
                dtype == core::DType::float16 ? "ormi_half_value(" + expected + ")" : expected);
        }
        loop.strides[1] = {0, 0, 0, 0, 0};
        EXPECT_EQ(codegen::input_value(core::DType::float32, input, loop), "input1[0]");
    }

    TEST(CPUJITInputTest, packedStridedLoadsUseThePlannedInnerDimension) {
        const fusion::Input input{1, {}, false};
        jit::Loop loop;
        loop.strided = true;
        loop.strides = {{}, {195, 0, 1, 65}};
        loop.reduced_dimensions = {0, 2};
        EXPECT_EQ(codegen::packed_input(input, core::DType::float32, loop),
            "_mm256_loadu_ps(input1 + index0 * 195 + inner_index + index3 * 65)");
        loop.strides[1][2] = 0;
        EXPECT_EQ(codegen::packed_input(input, core::DType::float32, loop),
            "_mm256_set1_ps(input1[index0 * 195 + index3 * 65])");
        loop.strides[1] = {0, 0, 0, 0};
        EXPECT_EQ(codegen::packed_input(input, core::DType::float32, loop), "_mm256_set1_ps(input1[0])");
    }

    TEST(CPUJITInputTest, retainsPackedAccessRejections) {
        const fusion::Input input{0, {}, true};
        jit::Loop loop;
        loop.inputs = {jit::Access::indexed};
        EXPECT_THROW(codegen::packed_input(input, core::DType::float32, loop), ormi::OperationError);
        loop.inputs[0] = static_cast<jit::Access>(999);
        EXPECT_THROW(codegen::packed_input(input, core::DType::float32, loop), ormi::OperationError);
        loop.strided = true;
        loop.strides = {{2}};
        loop.reduced_dimensions = {0};
        EXPECT_THROW(codegen::packed_input(input, core::DType::float16, loop), ormi::OperationError);
        EXPECT_THROW(codegen::packed_input(input, core::DType::float64, loop), ormi::OperationError);
        EXPECT_THROW(codegen::packed_input(input, core::DType::float32, loop), ormi::OperationError);
    }
}
