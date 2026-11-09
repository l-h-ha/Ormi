#include <array>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/codegen.h"
#include "backends/cpu/jit/codegen/support.h"
#include "core/runtime/fusion/expr.h"

namespace {
    namespace core = ormi::core;
    namespace jit = ormi::backends::cpu::jit;

    std::shared_ptr<core::Tensor> graph(core::DType dtype) {
        const auto input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{65}, dtype
        );
        return core::ops::sin(core::ops::add(core::ops::mul(input, input), input));
    }

    std::size_t occurrences(std::string_view source, std::string_view text) {
        std::size_t count = 0;
        std::size_t position = 0;
        while ((position = source.find(text, position)) != std::string_view::npos) {
            ++count;
            position += text.size();
        }
        return count;
    }

    constexpr std::string_view exports =
        "\n"
        "#if defined(_WIN32)\n"
        "#define ORMI_EXPORT __declspec(dllexport)\n"
        "#else\n"
        "#define ORMI_EXPORT __attribute__((visibility(\"default\")))\n"
        "#endif\n\n";

    TEST(CPUJITSourceTest, preservesScalarHeaderBytes) {
        std::ostringstream source;
        jit::codegen::emit_headers(source, false, false, false, false);
        EXPECT_EQ(source.str(), exports);
    }

    TEST(CPUJITSourceTest, emitsOnlyRequestedHeadersInOrder) {
        for (unsigned int flags = 0; flags < 16; ++flags) {
            SCOPED_TRACE(flags);
            const bool float16 = (flags & 1) != 0;
            const bool transcendental = (flags & 2) != 0;
            const bool intrinsics = (flags & 4) != 0;
            const bool vectorized = (flags & 8) != 0;
            std::ostringstream source;
            jit::codegen::emit_headers(source, float16, transcendental, intrinsics, vectorized);
            const auto generated = source.str();
            std::string expected;
            if (float16) {
                expected += "#include <bit>\n#include <cstdint>\n";
            }
            if (transcendental) {
                expected += "#include <cmath>\n";
            }
            if (intrinsics) {
                expected +=
                    "#if defined(ORMI_USE_F16C) || defined(ORMI_USE_VECMATH) || defined(ORMI_USE_AVX2)\n"
                    "#include <immintrin.h>\n#endif\n";
            }
            expected += exports;
            EXPECT_EQ(generated.substr(0, expected.size()), expected);
            if (!vectorized) {
                EXPECT_EQ(generated, expected);
                continue;
            }
            EXPECT_EQ(generated.find("#if defined(__clang__)\n"), expected.size());
            EXPECT_EQ(occurrences(generated, "inline T* ormi_assume_aligned("), 1);
            EXPECT_NE(generated.find("#elif defined(_MSC_VER)\n"), std::string::npos);
            EXPECT_NE(generated.find("#elif defined(__GNUC__)\n"), std::string::npos);
        }
    }

    TEST(CPUJITSourceTest, sharesSupportBeforeMixedDtypeBatchKernels) {
        const std::array roots{
            graph(core::DType::float64),
            graph(core::DType::float32),
            graph(core::DType::float16)
        };
        const std::array expressions{
            core::runtime::fusion::build(*roots[0]),
            core::runtime::fusion::build(*roots[1]),
            core::runtime::fusion::build(*roots[2])
        };
        for (const auto& expression : expressions) {
            ASSERT_TRUE(expression.has_value());
        }
        const std::array pointers{&*expressions[0], &*expressions[1], &*expressions[2]};
        const std::array loops{
            jit::plan(*expressions[0]), jit::plan(*expressions[1]), jit::plan(*expressions[2])
        };
        const auto source = jit::emit(pointers, loops);
        EXPECT_EQ(occurrences(source, "#include <bit>\n"), 1);
        EXPECT_EQ(occurrences(source, "#include <cmath>\n"), 1);
        EXPECT_EQ(occurrences(source, "#include <immintrin.h>\n"), 1);
        EXPECT_EQ(occurrences(source, "std::uint16_t ormi_half_bits(float value) noexcept {"), 1);
        EXPECT_EQ(occurrences(source, "#if defined(ORMI_USE_DIRECT_VECMATH)\n"), 1);
        const auto first = source.find("extern \"C\" ORMI_EXPORT void ormi_kernel_0(");
        const auto second = source.find("extern \"C\" ORMI_EXPORT void ormi_kernel_1(");
        const auto third = source.find("extern \"C\" ORMI_EXPORT void ormi_kernel_2(");
        ASSERT_NE(first, std::string::npos);
        ASSERT_NE(second, std::string::npos);
        ASSERT_NE(third, std::string::npos);
        EXPECT_LT(first, second);
        EXPECT_LT(second, third);
        EXPECT_LT(source.find("std::uint16_t ormi_half_bits("), first);
        EXPECT_LT(source.find("#if defined(ORMI_USE_DIRECT_VECMATH)\n"), first);
    }

    TEST(CPUJITSourceTest, preservesSingleAndBatchSymbolConventions) {
        for (const auto dtype : {core::DType::float16, core::DType::float32, core::DType::float64}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const auto root = graph(dtype);
            const auto expression = core::runtime::fusion::build(*root);
            ASSERT_TRUE(expression.has_value());
            const auto loop = jit::plan(*expression);
            const auto single = jit::emit(*expression, loop);
            const std::array expressions{&*expression};
            const std::array loops{loop};
            auto batch = jit::emit(expressions, loops);
            const auto position = batch.find("ormi_kernel_0(");
            ASSERT_NE(position, std::string::npos);
            batch.replace(position, std::string_view("ormi_kernel_0").size(), "ormi_kernel");
            EXPECT_EQ(batch, single);
        }
    }

    TEST(CPUJITSourceTest, keepsDuplicateBatchFunctionsInRequestOrder) {
        const auto root = graph(core::DType::float16);
        const auto expression = core::runtime::fusion::build(*root);
        ASSERT_TRUE(expression.has_value());
        const auto loop = jit::plan(*expression);
        std::array<const core::runtime::fusion::Expr*, 12> expressions;
        std::array<jit::Loop, 12> loops;
        expressions.fill(&*expression);
        loops.fill(loop);
        const auto source = jit::emit(expressions, loops);
        EXPECT_EQ(occurrences(source, "std::uint16_t ormi_half_bits(float value) noexcept {"), 1);
        EXPECT_EQ(occurrences(source, "extern \"C\" ORMI_EXPORT void ormi_kernel_"), 12);
        std::size_t position = 0;
        for (std::size_t i = 0; i < expressions.size(); ++i) {
            const auto next = source.find("ormi_kernel_" + std::to_string(i) + "(", position);
            ASSERT_NE(next, std::string::npos);
            position = next + 1;
        }
    }

    TEST(CPUJITSourceTest, rejectsInvalidBatchesAndExpressions) {
        const std::span<const core::runtime::fusion::Expr* const> empty_expressions;
        const std::span<const jit::Loop> empty_loops;
        EXPECT_THROW((void)jit::emit(empty_expressions, empty_loops), ormi::OperationError);
        const auto root = graph(core::DType::float32);
        const auto expression = core::runtime::fusion::build(*root);
        ASSERT_TRUE(expression.has_value());
        const std::array expressions{&*expression};
        EXPECT_THROW((void)jit::emit(expressions, {}), ormi::OperationError);
        const std::array<const core::runtime::fusion::Expr*, 1> missing{nullptr};
        const std::array loops{jit::plan(*expression)};
        EXPECT_THROW((void)jit::emit(missing, loops), ormi::OperationError);
        auto malformed = *expression;
        malformed.output = static_cast<core::runtime::fusion::Index>(malformed.nodes.size());
        EXPECT_THROW((void)jit::emit(malformed, loops[0]), ormi::OperationError);
    }
}
