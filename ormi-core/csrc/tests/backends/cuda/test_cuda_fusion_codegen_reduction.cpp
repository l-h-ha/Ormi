#include <memory>
#include <string>
#include <variant>

#include <gtest/gtest.h>

#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/codegen.h"
#include "backends/cuda/fusion/tile.h"
#include "core/runtime/fusion/expr.h"

namespace {
    namespace core = ormi::core;
    namespace fusion = ormi::backends::cuda::fusion;

    void reject_operation(const core::runtime::fusion::Expr& expression, const char* message) {
        try {
            (void)fusion::emit(expression);
            FAIL() << "Malformed fusion expression was accepted.";
        } catch (const ormi::OperationError& error) {
            EXPECT_EQ(std::string(error.what()), message);
        }
    }

    TEST(CUDAFusionCodegenReductionTest, tiledSumAndMeanPreserveFloatingPointRounding) {
        for (const auto dtype : {core::DType::float16, core::DType::float32, core::DType::float64}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const auto input = std::make_shared<core::Tensor>(
                core::Device::cpu(), core::Shape{2, 3, 257}, dtype
            );
            const auto mapped = core::ops::sin(core::ops::add(core::ops::mul(input, input), input));
            for (const bool mean : {false, true}) {
                SCOPED_TRACE(mean);
                const auto root = mean ? core::ops::mean(mapped, {0, 1}) : core::ops::sum(mapped, {0, 1});
                const auto expression = core::runtime::fusion::build(*root);
                ASSERT_TRUE(expression);
                ASSERT_TRUE(fusion::tile(*expression));
                const auto source = fusion::emit(*expression);
                const std::string scalar = dtype == core::DType::float64 ? "double" : "float";
                const std::string storage = dtype == core::DType::float16 ? "unsigned short" : scalar;
                EXPECT_NE(source.find(storage + "* output"), std::string::npos);
                EXPECT_NE(source.find("reduced_index < 6ull"), std::string::npos);
                EXPECT_EQ(source.find("ormi_finish"), std::string::npos);

                const auto* unary = std::get_if<core::runtime::fusion::Unary>(
                    &expression->nodes[expression->output].instruction
                );
                ASSERT_NE(unary, nullptr);
                auto operation = std::string(dtype == core::DType::float64 ? "sin(" : "sinf(") +
                    "v" + std::to_string(unary->input) + ")";
                if (dtype == core::DType::float16) {
                    operation = "ormi_half_round(" + operation + ")";
                }
                EXPECT_NE(source.find("const " + scalar + " v" +
                    std::to_string(expression->output) + " = " + operation + ";"), std::string::npos);

                auto result = mean ? "total / static_cast<" + scalar + ">(6ull)" : std::string("total");
                if (dtype == core::DType::float16) {
                    result = "ormi_half_bits(" + result + ")";
                }
                EXPECT_NE(source.find(" = " + result + ";\n"), std::string::npos);
            }
        }
    }

    TEST(CUDAFusionCodegenReductionTest, tiledReductionPreservesMultipleOuterCoordinatesAndTail) {
        const auto input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{2, 3, 5, 257}, core::DType::float32
        );
        const auto mapped = core::ops::add(core::ops::mul(input, input), input);
        const auto root = core::ops::sum(mapped, {1});
        const auto expression = core::runtime::fusion::build(*root);
        ASSERT_TRUE(expression);
        const auto tiled = fusion::tile(*expression);
        ASSERT_TRUE(tiled);
        ASSERT_EQ(tiled->axis, 3u);
        const auto source = fusion::emit(*expression);
        EXPECT_NE(source.find("const unsigned long long index0 = outer_index / 5ull;"), std::string::npos);
        EXPECT_NE(source.find("const unsigned long long index2 = outer_index % 5ull;"), std::string::npos);
        EXPECT_NE(source.find("const unsigned long long index1 = reduced_index;"), std::string::npos);
        EXPECT_NE(source.find("if (index3 >= 257ull)"), std::string::npos);
        EXPECT_NE(source.find("output[index0 * 1285 + index2 * 257 + index3]"), std::string::npos);
    }

    TEST(CUDAFusionCodegenReductionTest, preservesValidationPrecedence) {
        const auto input = std::make_shared<core::Tensor>(
            core::Device::cpu(), core::Shape{2, 257}, core::DType::float32
        );
        const auto mapped = core::ops::add(core::ops::mul(input, input), input);
        const auto root = core::ops::sum(mapped, {0});
        auto built = core::runtime::fusion::build(*root);
        ASSERT_TRUE(built);
        auto expression = *built;
        expression.dtype = core::DType::boolean;
        expression.reduction->op = static_cast<core::ops::ReduceOp>(255);
        expression.auxiliary_outputs = {expression.output};
        expression.output = static_cast<core::runtime::fusion::Index>(expression.nodes.size());
        reject_operation(expression, "Cannot generate CUDA code for a malformed fusion expression.");
        expression.output = built->output;
        reject_operation(expression, "CUDA fusion does not support auxiliary reduction outputs.");
        expression.auxiliary_outputs.clear();
        reject_operation(expression, "CUDA fusion received an unsupported reduction.");
        expression.reduction->op = core::ops::ReduceOp::sum;
        EXPECT_THROW((void)fusion::emit(expression), ormi::DTypeError);
    }
}
