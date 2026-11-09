#include <array>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <ormi/backend/fusion.h>
#include <ormi/core/error.h>

#include "backends/cuda/fusion/codegen.h"

namespace {
    namespace core = ormi::core;
    namespace fusion = core::runtime::fusion;
    namespace cuda = ormi::backends::cuda::fusion;

    fusion::Expr expression(fusion::Instruction instruction, core::DType dtype = core::DType::float32) {
        return {
            .dtype = dtype,
            .shape = {4},
            .nodes = {{dtype, fusion::Constant::one}, {dtype, std::move(instruction)}},
            .inputs = {},
            .output = 1,
            .auxiliary_outputs = {},
            .operations = 1,
            .elements = 4,
            .reduction = std::nullopt
        };
    }

    void reject_operation(const fusion::Expr& value, const char* message) {
        try {
            (void)cuda::emit(value);
            FAIL() << "Code generation accepted an invalid instruction.";
        } catch (const ormi::OperationError& error) {
            EXPECT_STREQ(error.what(), message);
        }
    }

    TEST(CUDAFusionCodegenValueTest, emitsFloatingPointOperationsAndConstants) {
        const std::array unaries{
            std::pair{core::ops::UnaryOp::neg, "-"},
            std::pair{core::ops::UnaryOp::exp, "exp"},
            std::pair{core::ops::UnaryOp::log, "log"},
            std::pair{core::ops::UnaryOp::sin, "sin"},
            std::pair{core::ops::UnaryOp::cos, "cos"}
        };
        const std::array binaries{
            std::pair{core::ops::BinaryOp::add, "+"},
            std::pair{core::ops::BinaryOp::sub, "-"},
            std::pair{core::ops::BinaryOp::mul, "*"},
            std::pair{core::ops::BinaryOp::div, "/"}
        };
        for (const auto dtype : {core::DType::float16, core::DType::float32, core::DType::float64}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const std::string type = dtype == core::DType::float64 ? "double" : "float";
            const std::string suffix = dtype == core::DType::float64 ? "" : "f";
            for (const auto& [op, name] : unaries) {
                auto value = expression(fusion::Unary{op, 0}, dtype);
                const auto text = op == core::ops::UnaryOp::neg
                    ? std::string("-v0") : name + suffix + "(v0)";
                EXPECT_NE(cuda::emit(value).find("const " + type + " v1 = " + text + ";"),
                    std::string::npos);
            }
            for (const auto& [op, token] : binaries) {
                auto value = expression(fusion::Binary{op, 0, 0}, dtype);
                EXPECT_NE(cuda::emit(value).find("const " + type + " v1 = v0 " + token + " v0;"),
                    std::string::npos);
            }
            for (const auto constant : {fusion::Constant::zero, fusion::Constant::one}) {
                auto value = expression(fusion::Unary{core::ops::UnaryOp::neg, 0}, dtype);
                value.nodes.front().instruction = constant;
                const auto literal = std::string(constant == fusion::Constant::zero ? "0.0" : "1.0");
                EXPECT_NE(cuda::emit(value).find("const " + type + " v0 = " + literal + suffix + ";"),
                    std::string::npos);
            }
        }
    }

    TEST(CUDAFusionCodegenValueTest, preservesFloat16RoundingWithGroupedOutputs) {
        auto value = expression(fusion::Binary{core::ops::BinaryOp::mul, 0, 0}, core::DType::float16);
        value.nodes.push_back({core::DType::float16, fusion::Unary{core::ops::UnaryOp::sin, 1}});
        value.output = 2;
        value.auxiliary_outputs = {1};
        value.operations = 2;
        const auto source = cuda::emit(value);
        EXPECT_NE(source.find("const float v1 = ormi_half_round(v0 * v0);"), std::string::npos);
        EXPECT_NE(source.find("const float v2 = sinf(v1);"), std::string::npos);
        EXPECT_NE(source.find("output[i] = ormi_half_bits(v2);"), std::string::npos);
        EXPECT_NE(source.find("output1[i] = ormi_half_bits(v1);"), std::string::npos);
    }

    TEST(CUDAFusionCodegenValueTest, rejectsSelfAndForwardReferencesBeforeFormatting) {
        const auto unknown_unary = static_cast<core::ops::UnaryOp>(255);
        const auto unknown_binary = static_cast<core::ops::BinaryOp>(255);
        for (const fusion::Index reference : {1u, 2u, std::numeric_limits<fusion::Index>::max()}) {
            SCOPED_TRACE(reference);
            const std::array<fusion::Instruction, 4> instructions{
                fusion::Cast{reference}, fusion::Unary{unknown_unary, reference},
                fusion::Binary{unknown_binary, reference, 0},
                fusion::Binary{unknown_binary, 0, reference}
            };
            for (const auto& instruction : instructions) {
                auto value = expression(instruction);
                value.nodes.back().dtype = core::DType::int32;
                reject_operation(value, "Fusion expression nodes must reference prior nodes.");
            }
        }
    }

    TEST(CUDAFusionCodegenValueTest, rejectsUnknownOperationsAfterValidReferences) {
        reject_operation(expression(fusion::Unary{static_cast<core::ops::UnaryOp>(255), 0}),
            "CUDA fusion received an unknown unary operation.");
        reject_operation(expression(fusion::Binary{static_cast<core::ops::BinaryOp>(255), 0, 0}),
            "CUDA fusion received an unknown binary operation.");
    }

    TEST(CUDAFusionCodegenValueTest, rejectsUnsupportedCastTypesAfterValidReferences) {
        for (const auto dtype : {core::DType::int32, core::DType::boolean, static_cast<core::DType>(255)}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            auto value = expression(fusion::Cast{0});
            value.nodes.back().dtype = dtype;
            try {
                (void)cuda::emit(value);
                FAIL() << "Code generation accepted an unsupported cast dtype.";
            } catch (const ormi::DTypeError& error) {
                EXPECT_STREQ(error.what(), dtype == static_cast<core::DType>(255)
                    ? "CUDA fusion received an unknown cast dtype."
                    : "CUDA fusion cannot cast to a non-floating dtype.");
            }
        }
    }
}
