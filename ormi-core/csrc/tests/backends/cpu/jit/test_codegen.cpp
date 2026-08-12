#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <gtest/gtest.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/codegen.h"
#include "core/runtime/fusion/expr.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> tensor(
        ormi::core::DType dtype,
        ormi::core::Shape shape = {4}
    ) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            std::move(shape),
            dtype
        );
    }

    TEST(CPUJITCodegenTest, emitsDeterministicFloat32Kernel) {
        const auto x = tensor(ormi::core::DType::float32);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(x, x), x)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto first = ormi::backends::cpu::jit::emit(*expression);
        const auto second = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_EQ(first, second);
        EXPECT_NE(first.find("extern \"C\" ORMI_EXPORT void ormi_kernel"), std::string::npos);
        EXPECT_NE(first.find("unsigned long long begin"), std::string::npos);
        EXPECT_NE(first.find("unsigned long long end"), std::string::npos);
        EXPECT_NE(
            first.find("for (unsigned long long i = begin; i < end; ++i)"),
            std::string::npos
        );
        EXPECT_NE(first.find("auto* output = static_cast<float*>(output_ptrs[0])"), std::string::npos);
        EXPECT_NE(first.find("const float v1 = v0 * v0;"), std::string::npos);
        EXPECT_NE(first.find("const float v2 = v1 + v0;"), std::string::npos);
        EXPECT_NE(first.find("const float v3 = std::sin(v2);"), std::string::npos);
        EXPECT_NE(first.find("output[i] = v3;"), std::string::npos);
        EXPECT_NE(first.find("#include <cmath>"), std::string::npos);
        EXPECT_EQ(first.find("#include <bit>"), std::string::npos);
        EXPECT_EQ(first.find("#include <cstdint>"), std::string::npos);
        EXPECT_EQ(first.find("#include <immintrin.h>"), std::string::npos);
        EXPECT_EQ(first.find("#include <memory>"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsFloat64Kernel) {
        const auto x = tensor(ormi::core::DType::float64);
        const auto output = ormi::core::ops::cos(ormi::core::ops::exp(x));
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("auto* output = static_cast<double*>(output_ptrs[0])"), std::string::npos);
        EXPECT_NE(source.find("const double v1 = std::exp(v0);"), std::string::npos);
        EXPECT_NE(source.find("const double v2 = std::cos(v1);"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsPackedFloat32MathWithScalarTail) {
        const auto x = tensor(ormi::core::DType::float32, {65});
        const auto output = ormi::core::ops::mul(
            ormi::core::ops::cos(x),
            x
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("defined(ORMI_USE_VECMATH)"), std::string::npos);
        EXPECT_NE(
            source.find("defined(ORMI_USE_DIRECT_VECMATH)"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("ormi_math8(math_ptr, 3, p0)"),
            std::string::npos
        );
        EXPECT_NE(source.find("i + 8 <= end"), std::string::npos);
        EXPECT_NE(source.find("const float v1 = std::cos(v0)"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsAuxiliaryOutputPointerAndStores) {
        const auto x = tensor(ormi::core::DType::float32, {65});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(x, x),
            x
        );
        const auto sine = ormi::core::ops::sin(mapped);
        const auto cosine = ormi::core::ops::cos(mapped);
        const std::array<ormi::core::Tensor*, 1> auxiliaries{cosine.get()};
        const std::unordered_set<const ormi::core::Tensor*> boundaries;
        const auto expression = ormi::core::runtime::fusion::build(
            *sine,
            auxiliaries,
            boundaries
        );
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("auto* ORMI_RESTRICT output1"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("static_cast<float*>(output_ptrs[1])"),
            std::string::npos
        );
        EXPECT_NE(source.find("output[i] = v3;"), std::string::npos);
        EXPECT_NE(source.find("output1[i] = v4;"), std::string::npos);
        EXPECT_NE(
            source.find("ormi_sincos8(math_ptr, p2, p3, p4);"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("ormi_sincos8_result (__vectorcall *)(__m256)"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, marksContiguousArithmeticForVectorization) {
        const auto x = tensor(ormi::core::DType::float32, {16});
        const auto square = ormi::core::ops::mul(x, x);
        const auto added = ormi::core::ops::add(square, x);
        const auto output = ormi::core::ops::mul(added, x);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_EQ(source.find("#include <bit>"), std::string::npos);
        EXPECT_EQ(source.find("#include <cmath>"), std::string::npos);
        EXPECT_EQ(source.find("#include <cstdint>"), std::string::npos);
        EXPECT_EQ(source.find("#include <immintrin.h>"), std::string::npos);
        EXPECT_EQ(source.find("#include <memory>"), std::string::npos);
        EXPECT_NE(source.find("#define ORMI_RESTRICT"), std::string::npos);
        EXPECT_NE(source.find("#define ORMI_VECTORIZE_LOOP"), std::string::npos);
        EXPECT_NE(source.find("inline T* ormi_assume_aligned"), std::string::npos);
        EXPECT_NE(source.find("auto* ORMI_RESTRICT output"), std::string::npos);
        EXPECT_NE(source.find("ormi_assume_aligned("), std::string::npos);
        EXPECT_EQ(source.find("std::assume_aligned<64>"), std::string::npos);
        EXPECT_NE(source.find("    ORMI_VECTORIZE_LOOP\n    for"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, hoistsScalarBroadcastsFromVectorizedLoop) {
        const auto input = tensor(ormi::core::DType::float32, {16});
        const auto scale = tensor(ormi::core::DType::float32, {});
        const auto bias = tensor(ormi::core::DType::float32, {});
        const auto scaled = ormi::core::ops::mul(input, scale);
        const auto shifted = ormi::core::ops::add(scaled, bias);
        const auto output = ormi::core::ops::mul(shifted, input);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);
        const auto loop = source.find("    ORMI_VECTORIZE_LOOP\n    for");

        ASSERT_NE(loop, std::string::npos);
        EXPECT_LT(source.find("const float input1_scalar = input1[0];"), loop);
        EXPECT_LT(source.find("const float input2_scalar = input2[0];"), loop);
        EXPECT_NE(source.find("const float v1 = input1_scalar;"), std::string::npos);
        EXPECT_NE(source.find("const float v3 = input2_scalar;"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsVectorizedRowBroadcastSpan) {
        const auto lhs = tensor(ormi::core::DType::float32, {2, 4});
        const auto rhs = tensor(ormi::core::DType::float32, {4});
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("base = begin; base < end; base += 4"),
            std::string::npos
        );
        EXPECT_NE(source.find("const auto input1_base = 0;"), std::string::npos);
        EXPECT_NE(source.find("j < 4; ++j"), std::string::npos);
        EXPECT_NE(source.find("const auto i = base + j;"), std::string::npos);
        EXPECT_NE(source.find("input1[input1_base + j]"), std::string::npos);
        EXPECT_NE(source.find("        ORMI_VECTORIZE_LOOP\n"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsVectorizedColumnBroadcastSpan) {
        const auto lhs = tensor(ormi::core::DType::float32, {2, 4});
        const auto rhs = tensor(ormi::core::DType::float32, {2, 1});
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("base = begin; base < end; base += 4"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("const float input1_fixed = input1[((base / 4) % 2)];"),
            std::string::npos
        );
        EXPECT_NE(source.find("const float v2 = input1_fixed;"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsArbitraryRankBroadcastBase) {
        const auto lhs = tensor(ormi::core::DType::float32, {2, 3, 4});
        const auto rhs = tensor(ormi::core::DType::float32, {3, 4});
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("base = begin; base < end; base += 4"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("const auto input1_base = ((base / 4) % 3) * 4;"),
            std::string::npos
        );
        EXPECT_NE(source.find("input1[input1_base + j]"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsBroadcastInputOffsets) {
        const auto lhs = tensor(ormi::core::DType::float32, {2, 1});
        const auto rhs = tensor(ormi::core::DType::float32, {3});
        const auto output = ormi::core::ops::sin(ormi::core::ops::add(lhs, rhs));
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("input0[((i / 3) % 2)]"), std::string::npos);
        EXPECT_NE(source.find("input1[(i % 3)]"), std::string::npos);
        EXPECT_EQ(source.find("ORMI_VECTORIZE_LOOP"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsSafeOffsetsForEmptyBroadcast) {
        const auto lhs = tensor(ormi::core::DType::float32, {0, 3});
        const auto rhs = tensor(ormi::core::DType::float32, {3});
        const auto output = ormi::core::ops::sin(ormi::core::ops::add(lhs, rhs));
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_EQ(source.find("% 0"), std::string::npos);
        EXPECT_NE(source.find("input1[0]"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsNestedStridedTraversal) {
        const auto base = tensor(ormi::core::DType::float32, {2, 3});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(transposed, transposed)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("index0 < 3"), std::string::npos);
        EXPECT_NE(source.find("index1 < 2"), std::string::npos);
        EXPECT_NE(
            source.find("input0[index0 + index1 * 3]"),
            std::string::npos
        );
        EXPECT_EQ(source.find("i / 2"), std::string::npos);
        EXPECT_EQ(source.find("i % 2"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, tilesLargeTwoDimensionalTranspose) {
        const auto base = tensor(ormi::core::DType::float32, {64, 64});
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(transposed, transposed)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("tile0 += 32"), std::string::npos);
        EXPECT_NE(source.find("tile1 += 32"), std::string::npos);
        EXPECT_NE(
            source.find("const auto i = index0 * 64 + index1;"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsFloat16KernelWithNodeRounding) {
        const auto x = tensor(ormi::core::DType::float16);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(x, x), x)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("std::uint16_t*>(output_ptrs[0])"), std::string::npos);
        EXPECT_NE(source.find("ormi_half_value(input0[i])"), std::string::npos);
        EXPECT_NE(source.find("ormi_half_round(v0 * v0)"), std::string::npos);
        EXPECT_NE(source.find("ormi_half_round(v1 + v0)"), std::string::npos);
        EXPECT_NE(source.find("const float v3 = std::sin(v2)"), std::string::npos);
        EXPECT_NE(source.find("output[i] = ormi_half_bits(v3)"), std::string::npos);
        EXPECT_NE(source.find("#include <bit>"), std::string::npos);
        EXPECT_NE(source.find("#include <cmath>"), std::string::npos);
        EXPECT_NE(source.find("#include <cstdint>"), std::string::npos);
        EXPECT_NE(source.find("#include <immintrin.h>"), std::string::npos);
        EXPECT_EQ(source.find("#include <memory>"), std::string::npos);
        EXPECT_EQ(source.find("ORMI_VECTORIZE_LOOP"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsPackedFloat16ArithmeticWithTail) {
        const auto x = tensor(ormi::core::DType::float16, {65});
        const auto square = ormi::core::ops::mul(x, x);
        const auto added = ormi::core::ops::add(square, x);
        const auto output = ormi::core::ops::mul(added, x);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("#if defined(ORMI_USE_F16C)"), std::string::npos);
        EXPECT_NE(source.find("_mm256_cvtph_ps"), std::string::npos);
        EXPECT_NE(
            source.find("const __m256 p1 = ormi_half_round8(_mm256_mul_ps(p0, p0));"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("const __m256 p2 = ormi_half_round8(_mm256_add_ps(p1, p0));"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("const __m256 p3 = _mm256_mul_ps(p2, p0);"),
            std::string::npos
        );
        EXPECT_NE(source.find("i + 8 <= end"), std::string::npos);
        EXPECT_NE(source.find("ormi_half_bits8(p3)"), std::string::npos);
        EXPECT_NE(source.find("for (; i < end; ++i)"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsPackedFloat16ScalarBroadcast) {
        const auto input = tensor(ormi::core::DType::float16, {65});
        const auto scale = tensor(ormi::core::DType::float16, {});
        const auto scaled = ormi::core::ops::mul(input, scale);
        const auto shifted = ormi::core::ops::add(scaled, scale);
        const auto output = ormi::core::ops::mul(shifted, input);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("const std::uint16_t input1_scalar = input1[0];"),
            std::string::npos
        );
        EXPECT_NE(
            source.find(
                "_mm256_set1_ps(ormi_half_value(input1_scalar))"
            ),
            std::string::npos
        );
        EXPECT_NE(source.find("for (; i < end; ++i)"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsPackedFloat16RowBroadcast) {
        const auto lhs = tensor(ormi::core::DType::float16, {4, 17});
        const auto rhs = tensor(ormi::core::DType::float16, {17});
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("base = begin; base < end; base += 17"),
            std::string::npos
        );
        EXPECT_NE(source.find("const auto input1_base = 0;"), std::string::npos);
        EXPECT_NE(
            source.find("ormi_half_load8(input1 + input1_base + j)"),
            std::string::npos
        );
        EXPECT_NE(source.find("j + 8 <= 17"), std::string::npos);
        EXPECT_NE(source.find("for (; j < 17; ++j)"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsPackedFloat16ColumnBroadcast) {
        const auto lhs = tensor(ormi::core::DType::float16, {4, 17});
        const auto rhs = tensor(ormi::core::DType::float16, {4, 1});
        const auto square = ormi::core::ops::mul(lhs, lhs);
        const auto added = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::mul(added, lhs);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("const std::uint16_t input1_fixed = input1[((base / 17) % 4)];"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("_mm256_set1_ps(ormi_half_value(input1_fixed))"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsPackedFloat16LastAxisMean) {
        const auto input = tensor(ormi::core::DType::float16, {2, 65});
        const auto square = ormi::core::ops::mul(input, input);
        const auto added = ormi::core::ops::add(square, input);
        const auto mapped = ormi::core::ops::mul(added, input);
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("__m256 partials = _mm256_setzero_ps()"), std::string::npos);
        EXPECT_NE(source.find("reduced_index + 8 <= 65"), std::string::npos);
        EXPECT_NE(
            source.find("const __m256 p3 = ormi_half_round8(_mm256_mul_ps(p2, p0));"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("partials = _mm256_add_ps(partials, p3)"),
            std::string::npos
        );
        EXPECT_NE(source.find("_mm256_store_ps(lanes, partials)"), std::string::npos);
        EXPECT_NE(source.find("for (; reduced_index < 65"), std::string::npos);
        EXPECT_NE(
            source.find("output_index = begin; output_index < end"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("ormi_half_bits(total / static_cast<double>(65ull))"),
            std::string::npos
        );
        EXPECT_NE(source.find("#else\n    for"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsFusedSum) {
        const auto x = tensor(ormi::core::DType::float32);
        const auto mapped = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(x, x), x)
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("double total = 0.0f;"), std::string::npos);
        EXPECT_NE(source.find("float block_total = 0.0f;"), std::string::npos);
        EXPECT_NE(source.find("block_total += v3;"), std::string::npos);
        EXPECT_NE(source.find("if (++block_size == 32)"), std::string::npos);
        EXPECT_NE(source.find("total += block_total;"), std::string::npos);
        EXPECT_NE(
            source.find("output_index = 0; output_index < 1"),
            std::string::npos
        );
        EXPECT_NE(source.find("output[output_index] = total;"), std::string::npos);
        EXPECT_EQ(source.find("output[i] ="), std::string::npos);
        EXPECT_EQ(source.find("ORMI_VECTORIZE_LOOP"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsDirectMiddleAxisReduction) {
        const auto x = tensor(ormi::core::DType::float32, {2, 3, 4});
        const auto mapped = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(x, x), x)
        );
        const auto output = ormi::core::ops::sum(mapped, {1}, true);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("unsigned long long output_index = begin"),
            std::string::npos
        );
        EXPECT_NE(source.find("index0 = begin / 4"), std::string::npos);
        EXPECT_NE(source.find("index0 < end / 4"), std::string::npos);
        EXPECT_NE(source.find("index2 = 0"), std::string::npos);
        EXPECT_NE(source.find("index2 < 4"), std::string::npos);
        EXPECT_NE(source.find("index1 < 3"), std::string::npos);
        EXPECT_NE(
            source.find("input0[index0 * 12 + index1 * 4 + index2]"),
            std::string::npos
        );
        EXPECT_EQ(source.find("reduced_index"), std::string::npos);
        EXPECT_EQ(source.find("output_index /"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsPackedSeparatedAxesReductionWithFallback) {
        const auto input = tensor(
            ormi::core::DType::float32,
            {32, 16, 64}
        );
        const auto row = tensor(ormi::core::DType::float32, {64});
        const auto column = tensor(
            ormi::core::DType::float32,
            {16, 1}
        );
        const auto shifted = ormi::core::ops::add(
            ormi::core::ops::mul(input, row),
            column
        );
        const auto output = ormi::core::ops::mean(
            ormi::core::ops::mul(shifted, shifted),
            {0, 2}
        );
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("#include <immintrin.h>"), std::string::npos);
        EXPECT_NE(
            source.find("#if defined(ORMI_USE_AVX2)"),
            std::string::npos
        );
        EXPECT_NE(
            source.find(
                "_mm256_loadu_ps(input0 + index0 * 1024 + "
                "index1 * 64 + inner_index)"
            ),
            std::string::npos
        );
        EXPECT_NE(
            source.find("_mm256_loadu_ps(input1 + inner_index)"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("_mm256_set1_ps(input2[index1])"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("#else\n\n    unsigned long long output_index"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsTiledFirstAxisReduction) {
        const auto x = tensor(
            ormi::core::DType::float32,
            {33, 70}
        );
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(x, x),
            x
        );
        const auto output = ormi::core::ops::sum(mapped, {0});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("tile = begin; tile < end; tile += 32"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("tile + 32 < end ? 32 : end - tile"),
            std::string::npos
        );
        EXPECT_NE(source.find("totals[32]"), std::string::npos);
        EXPECT_NE(source.find("index0 < 33"), std::string::npos);
        EXPECT_NE(
            source.find("const auto index1 = tile + lane"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("input0[index0 * 70 + index1]"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("output[tile + lane] = totals[lane]"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsTiledTransposedLastAxisReduction) {
        const auto base = tensor(
            ormi::core::DType::float32,
            {70, 33}
        );
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            transposed
        );
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("tile = begin; tile < end; tile += 32"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("tile + 32 < end ? 32 : end - tile"),
            std::string::npos
        );
        EXPECT_NE(source.find("index1 < 70"), std::string::npos);
        EXPECT_NE(
            source.find("const auto index0 = tile + lane"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("input0[index0 + index1 * 33]"),
            std::string::npos
        );
        EXPECT_NE(
            source.find(
                "output[tile + lane] = "
                "totals[lane] / static_cast<float>(70ull)"
            ),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsTiledRankThreeMiddleAxisReduction) {
        const auto base = tensor(
            ormi::core::DType::float32,
            {64, 96, 32}
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(permuted, permuted),
            permuted
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find(
                "tile = begin / 96; tile < end / 96; tile += 16"
            ),
            std::string::npos
        );
        EXPECT_NE(source.find("index2 = 0; index2 < 96"), std::string::npos);
        EXPECT_NE(source.find("index1 = 0; index1 < 64"), std::string::npos);
        EXPECT_NE(
            source.find("const auto index0 = tile + lane"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("input0[index0 + index1 * 3072 + index2 * 32]"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("output[(tile + lane) * 96 + index2]"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsDirectStridesForTransposedReduction) {
        const auto base = tensor(
            ormi::core::DType::float32,
            {2, 3}
        );
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto rhs = tensor(ormi::core::DType::float32, {2});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            rhs
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("unsigned long long output_index = begin"),
            std::string::npos
        );
        EXPECT_NE(source.find("index0 = begin"), std::string::npos);
        EXPECT_NE(source.find("index0 < end"), std::string::npos);
        EXPECT_NE(source.find("index1 < 2"), std::string::npos);
        EXPECT_NE(
            source.find("input0[index0 + index1 * 3]"),
            std::string::npos
        );
        EXPECT_NE(source.find("input1[index1]"), std::string::npos);
        EXPECT_EQ(source.find("reduced_index"), std::string::npos);
        EXPECT_EQ(source.find("output_index /"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsPartialSlabsForGlobalTranspose) {
        const auto base = tensor(
            ormi::core::DType::float32,
            {2, 3}
        );
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            transposed
        );
        const auto output = ormi::core::ops::mean(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("index0 = begin"), std::string::npos);
        EXPECT_NE(source.find("index0 < end"), std::string::npos);
        EXPECT_NE(source.find("index1 = 0"), std::string::npos);
        EXPECT_NE(source.find("index1 < 2"), std::string::npos);
        EXPECT_LT(
            source.find("index1 = 0"),
            source.find("index0 = begin")
        );
        EXPECT_NE(
            source.find("input0[index0 + index1 * 3]"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("output[output_index] = total"),
            std::string::npos
        );
        EXPECT_EQ(source.find("static_cast<float>(6ull)"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsVectorizedGlobalTransposeWithScalarTail) {
        const auto base = tensor(
            ormi::core::DType::float32,
            {65, 65}
        );
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            transposed
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("index1 = begin"), std::string::npos);
        EXPECT_NE(source.find("index1 < end"), std::string::npos);
        EXPECT_NE(source.find("double total = 0.0f;"), std::string::npos);
        EXPECT_NE(source.find("float partials[8] = {}"), std::string::npos);
        EXPECT_NE(
            source.find("unsigned long long inner_index = 0"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("inner_index + 8 <= 65"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("const auto index0 = inner_index + lane"),
            std::string::npos
        );
        EXPECT_NE(source.find("ORMI_VECTORIZE_LOOP"), std::string::npos);
        EXPECT_NE(
            source.find("for (; inner_index < 65; ++inner_index)"),
            std::string::npos
        );
        EXPECT_NE(source.find("output[0] = total"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsTiledMixedLayoutGlobalReduction) {
        const auto base = tensor(
            ormi::core::DType::float32,
            {97, 65}
        );
        const auto transposed =
            ormi::core::ops::transpose(base, {1, 0});
        const auto contiguous = tensor(
            ormi::core::DType::float32,
            {65, 97}
        );
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(transposed, transposed),
            contiguous
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("tile1 = begin; tile1 < end; tile1 += 16"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("tile0 = 0; tile0 < 65; tile0 += 16"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("index1 = tile1; index1 < end1"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("index0 = tile0; index0 < end0"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("input0[index0 + index1 * 65]"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("input1[index0 * 97 + index1]"),
            std::string::npos
        );
        EXPECT_NE(source.find("output[0] = total"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsTiledRankThreeMixedGlobalReduction) {
        const auto base = tensor(
            ormi::core::DType::float32,
            {64, 96, 32}
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto contiguous = tensor(
            ormi::core::DType::float32,
            {32, 64, 96}
        );
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(permuted, permuted),
            contiguous
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("index1 = 0; index1 < 64"), std::string::npos);
        EXPECT_NE(
            source.find("tile2 = begin; tile2 < end; tile2 += 16"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("tile0 = 0; tile0 < 32; tile0 += 16"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("input0[index0 + index1 * 3072 + index2 * 32]"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("input1[index0 * 6144 + index1 * 96 + index2]"),
            std::string::npos
        );
        EXPECT_NE(source.find("output[0] = total"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsFusedMeanFinalization) {
        const auto x = tensor(ormi::core::DType::float32, {2, 3});
        const auto mapped = ormi::core::ops::add(ormi::core::ops::mul(x, x), x);
        const auto output = ormi::core::ops::mean(mapped, {0});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("block_total += v2;"), std::string::npos);
        EXPECT_NE(
            source.find("output[output_index] = total / static_cast<double>(2ull);"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsPackedAutocastFloat16ArithmeticSum) {
        const auto input = tensor(ormi::core::DType::float16, {65});
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            const auto square = ormi::core::ops::mul(input, input);
            output = ormi::core::ops::sum(
                ormi::core::ops::add(square, input)
            );
        }
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("double total = 0.0;"), std::string::npos);
        EXPECT_NE(source.find("reduced_index + 256 <= end"), std::string::npos);
        EXPECT_NE(source.find("block < 32"), std::string::npos);
        EXPECT_NE(source.find("block * 8"), std::string::npos);
        EXPECT_NE(
            source.find("const __m256 p1 = ormi_half_round8(_mm256_mul_ps(p0, p0));"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("const __m256 p2 = ormi_half_round8(_mm256_add_ps(p1, p0));"),
            std::string::npos
        );
        EXPECT_NE(source.find("const __m256 p3 = p2;"), std::string::npos);
        EXPECT_NE(
            source.find("partials = _mm256_add_ps(partials, p3);"),
            std::string::npos
        );
        EXPECT_NE(source.find("output[output_index] = total;"), std::string::npos);
        EXPECT_EQ(
            source.find("output[output_index] = ormi_half_bits(total);"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsPackedFloat32Float16RoundTrip) {
        const auto input = tensor(ormi::core::DType::float32, {65});
        const auto narrowed = ormi::core::ops::cast(
            input,
            ormi::core::DType::float16
        );
        const auto square = ormi::core::ops::mul(narrowed, narrowed);
        const auto output = ormi::core::ops::cast(
            square,
            ormi::core::DType::float32
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("const __m256 p0 = _mm256_loadu_ps(input0 + i);"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("const __m256 p1 = ormi_half_round8(p0);"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("const __m256 p2 = ormi_half_round8(_mm256_mul_ps(p1, p1));"),
            std::string::npos
        );
        EXPECT_NE(source.find("const __m256 p3 = p2;"), std::string::npos);
        EXPECT_NE(
            source.find("_mm256_storeu_ps(output + i, p3);"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsVectorizedArithmeticGlobalSum) {
        const auto x = tensor(ormi::core::DType::float32, {64});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(x, x),
            x
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("const auto ormi_map"), std::string::npos);
        EXPECT_NE(source.find("double total = 0.0f;"), std::string::npos);
        EXPECT_NE(source.find("float partials[8] = {};"), std::string::npos);
        EXPECT_NE(
            source.find("unsigned long long reduced_index = begin"),
            std::string::npos
        );
        EXPECT_NE(source.find("reduced_index + 256 <= end"), std::string::npos);
        EXPECT_NE(source.find("block < 32"), std::string::npos);
        EXPECT_NE(source.find("block * 8 + lane"), std::string::npos);
        EXPECT_NE(source.find("reduced_index + 8 <= end"), std::string::npos);
        EXPECT_NE(source.find("partials[lane] += ormi_map("), std::string::npos);
        EXPECT_NE(
            source.find("output[output_index] = total"),
            std::string::npos
        );
        EXPECT_NE(source.find("ORMI_VECTORIZE_LOOP"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsVectorizedLastAxisMean) {
        const auto x = tensor(ormi::core::DType::float64, {2, 33});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(x, x),
            x
        );
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("output_index = begin; output_index < end"),
            std::string::npos
        );
        EXPECT_NE(source.find("double partials[4] = {};"), std::string::npos);
        EXPECT_NE(source.find("reduced_index + 4 <= 33"), std::string::npos);
        EXPECT_NE(
            source.find("output[output_index] = total / static_cast<double>(33ull);"),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, emitsTypedAutocastReduction) {
        const auto x = tensor(ormi::core::DType::float16);
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            output = ormi::core::ops::sum(ormi::core::ops::sin(x));
        }
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(
            source.find("const std::uint16_t*>(input_ptrs[0])"),
            std::string::npos
        );
        EXPECT_NE(source.find("const float v0 = ormi_half_value(input0[i]);"), std::string::npos);
        EXPECT_NE(source.find("const float v1 = static_cast<float>(v0);"), std::string::npos);
        EXPECT_NE(source.find("const float v2 = std::sin(v1);"), std::string::npos);
        EXPECT_NE(source.find("auto* output = static_cast<float*>(output_ptrs[0])"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsSeveralKernelsWithSharedSupportCode) {
        const auto x = tensor(ormi::core::DType::float32, {65});
        const auto first_output = ormi::core::ops::sin(
            ormi::core::ops::mul(x, x)
        );
        const auto second_output = ormi::core::ops::cos(
            ormi::core::ops::add(x, x)
        );
        const auto first = ormi::core::runtime::fusion::build(*first_output);
        const auto second = ormi::core::runtime::fusion::build(*second_output);
        ASSERT_TRUE(first.has_value());
        ASSERT_TRUE(second.has_value());

        const std::array<const ormi::core::runtime::fusion::Expr*, 2> expressions{
            &*first,
            &*second
        };
        const std::array<ormi::backends::cpu::jit::Loop, 2> loops{
            ormi::backends::cpu::jit::plan(*first),
            ormi::backends::cpu::jit::plan(*second)
        };
        const auto source = ormi::backends::cpu::jit::emit(expressions, loops);

        EXPECT_NE(
            source.find("extern \"C\" ORMI_EXPORT void ormi_kernel_0"),
            std::string::npos
        );
        EXPECT_NE(
            source.find("extern \"C\" ORMI_EXPORT void ormi_kernel_1"),
            std::string::npos
        );
        const auto math_table = source.find("struct ormi_math_table");
        ASSERT_NE(math_table, std::string::npos);
        const auto buffered_math_table = source.find(
            "struct ormi_math_table",
            math_table + 1
        );
        ASSERT_NE(buffered_math_table, std::string::npos);
        EXPECT_EQ(
            source.find("struct ormi_math_table", buffered_math_table + 1),
            std::string::npos
        );
    }

    TEST(CPUJITCodegenTest, rejectsMalformedExpression) {
        ormi::core::runtime::fusion::Expr expression{
            .dtype = ormi::core::DType::float32,
            .nodes = {},
            .inputs = {},
            .output = 0,
            .operations = 0,
            .elements = 0,
            .reduction = std::nullopt
        };

        EXPECT_THROW((void)ormi::backends::cpu::jit::emit(expression), ormi::OperationError);
    }
}
