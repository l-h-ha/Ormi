#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <ormi/core/autocast.h>
#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/float16.h>
#include <ormi/core/memory/buffer.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/codegen.h"
#include "backends/cuda/fusion/tile.h"
#include "core/runtime/fusion/expr.h"

namespace {
    TEST(CUDAFusionCodegenTest, emitsDeterministicFloat32Kernel) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto first = ormi::backends::cuda::fusion::emit(*expression);
        const auto second = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_EQ(first, second);
        EXPECT_NE(first.find("extern \"C\" __global__ void ormi_kernel"), std::string::npos);
        EXPECT_NE(first.find("const float v1 = v0 * v0;"), std::string::npos);
        EXPECT_NE(first.find("const float v2 = v1 + v0;"), std::string::npos);
        EXPECT_NE(first.find("const float v3 = sinf(v2);"), std::string::npos);
        EXPECT_NE(first.find("output[i] = v3;"), std::string::npos);
    }

    TEST(CUDAFusionCodegenTest, emitsGenericAuxiliaryOutputs) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        const auto shared = ormi::core::ops::add(input, input);
        const auto primary = ormi::core::ops::neg(shared);
        const auto auxiliary = ormi::core::ops::exp(shared);
        const std::array<ormi::core::Tensor*, 1> auxiliaries{
            auxiliary.get()
        };
        const std::unordered_set<const ormi::core::Tensor*> boundaries;
        const auto expression = ormi::core::runtime::fusion::build(
            *primary,
            auxiliaries,
            boundaries
        );
        ASSERT_TRUE(expression.has_value());
        ASSERT_EQ(expression->auxiliary_outputs.size(), 1u);

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_NE(
            source.find("float* output,\n    float* output1"),
            std::string::npos
        );
        EXPECT_NE(source.find("output[i] = v2;"), std::string::npos);
        EXPECT_NE(source.find("output1[i] = v3;"), std::string::npos);
    }

    TEST(CUDAFusionCodegenTest, emitsCompleteTypedKernelSignatures) {
        for (const auto dtype : {
            ormi::core::DType::float16,
            ormi::core::DType::float32,
            ormi::core::DType::float64
        }) {
            const auto lhs = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(), ormi::core::Shape{4}, dtype
            );
            const auto rhs = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(), ormi::core::Shape{4}, dtype
            );
            const auto mapped = ormi::core::ops::sin(
                ormi::core::ops::add(ormi::core::ops::mul(lhs, rhs), lhs)
            );
            const auto auxiliary = ormi::core::ops::cos(mapped);
            const std::array<ormi::core::Tensor*, 1> auxiliaries{auxiliary.get()};
            const auto matrix = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(), ormi::core::Shape{4, 1024}, dtype
            );
            const auto row = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(), ormi::core::Shape{1024}, dtype
            );
            const auto tiled = ormi::core::ops::sum(
                ormi::core::ops::add(ormi::core::ops::mul(matrix, matrix), row), {0}
            );
            const std::array roots{mapped, mapped, ormi::core::ops::sum(mapped), tiled};
            const std::string storage = dtype == ormi::core::DType::float16
                ? "unsigned short"
                : dtype == ormi::core::DType::float64 ? "double" : "float";
            const std::string scalar = dtype == ormi::core::DType::float64 ? "double" : "float";

            for (std::size_t layout = 0; layout < roots.size(); ++layout) {
                SCOPED_TRACE(static_cast<int>(dtype));
                SCOPED_TRACE(layout);
                const auto expression = layout == 1
                    ? ormi::core::runtime::fusion::build(*roots[layout], auxiliaries)
                    : ormi::core::runtime::fusion::build(*roots[layout]);
                ASSERT_TRUE(expression.has_value());
                ASSERT_EQ(expression->inputs.size(), 2u);
                ASSERT_EQ(ormi::backends::cuda::fusion::tile(*expression).has_value(), layout == 3);

                std::string expected = "extern \"C\" __global__ void ormi_kernel(\n    " +
                    (layout == 2 ? scalar : storage) + "* output";
                if (layout == 1) {
                    expected += ",\n    " + storage + "* output1";
                }
                expected += ",\n    const " + storage + "* input0";
                expected += ",\n    const " + storage + "* input1";
                expected += layout >= 2
                    ? ",\n    unsigned long long parts\n"
                    : ",\n    unsigned long long elements\n";
                expected += ") {\n";

                const auto source = ormi::backends::cuda::fusion::emit(*expression);
                const auto start = source.find("extern \"C\" __global__ void ormi_kernel(\n");
                ASSERT_NE(start, std::string::npos);
                const auto end = source.find(") {\n", start);
                ASSERT_NE(end, std::string::npos);
                EXPECT_EQ(source.substr(start, end + 4 - start), expected);
            }
        }
    }

    TEST(CUDAFusionCodegenTest, emitsMixedInputTypesInBindingOrder) {
        const auto lhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{4}, ormi::core::DType::float16
        );
        const auto rhs = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(), ormi::core::Shape{4}, ormi::core::DType::float64
        );
        const auto output = ormi::core::ops::add(ormi::core::ops::mul(lhs, rhs), rhs);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());
        ASSERT_EQ(expression->inputs.size(), 2u);
        ASSERT_EQ(expression->inputs[0], lhs.get());
        ASSERT_EQ(expression->inputs[1], rhs.get());

        const auto source = ormi::backends::cuda::fusion::emit(*expression);
        const std::string expected =
            "extern \"C\" __global__ void ormi_kernel(\n"
            "    double* output,\n"
            "    const unsigned short* input0,\n"
            "    const double* input1,\n"
            "    unsigned long long elements\n"
            ") {\n";
        EXPECT_NE(source.find(expected), std::string::npos);
    }

    TEST(CUDAFusionCodegenTest, rejectsInvalidAuxiliaryOutputIndex) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float32
        );
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(input, input)
        );
        auto expression = ormi::core::runtime::fusion::build(*output).value();
        expression.auxiliary_outputs.push_back(
            static_cast<ormi::core::runtime::fusion::Index>(
                expression.nodes.size()
            )
        );

        EXPECT_THROW(
            (void)ormi::backends::cuda::fusion::emit(expression),
            ormi::OperationError
        );
    }

    TEST(CUDAFusionCodegenTest, rejectsMalformedExpression) {
        const ormi::core::runtime::fusion::Expr malformed{
            .dtype = ormi::core::DType::float32,
            .nodes = {},
            .inputs = {},
            .output = 0,
            .operations = 0,
            .elements = 0,
            .reduction = std::nullopt
        };

        EXPECT_THROW(
            (void)ormi::backends::cuda::fusion::emit(malformed),
            ormi::OperationError
        );
    }

    TEST(CUDAFusionCodegenTest, emitsSelfContainedFloat16Kernel) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2},
            ormi::core::DType::float16
        );
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_EQ(source.find("#include <cuda_fp16.h>"), std::string::npos);
        EXPECT_NE(source.find("cvt.rn.f16.f32"), std::string::npos);
        EXPECT_NE(source.find("cvt.f32.f16"), std::string::npos);
        EXPECT_NE(
            source.find(
                "const float v0 = ormi_half_value(input0[i]);"
            ),
            std::string::npos
        );
        EXPECT_NE(source.find("const float v1 = ormi_half_round(v0 * v0);"), std::string::npos);
        EXPECT_NE(source.find("const float v2 = ormi_half_round(v1 + v0);"), std::string::npos);
        EXPECT_NE(
            source.find("const float v3 = sinf(v2);"),
            std::string::npos
        );
        EXPECT_NE(source.find("output[i] = ormi_half_bits(v3);"), std::string::npos);
    }

    TEST(CUDAFusionCodegenTest, emitsOnlyRequiredSupportRoutines) {
        for (const auto dtype : {
            ormi::core::DType::float16,
            ormi::core::DType::float32,
            ormi::core::DType::float64
        }) {
            const auto input = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(), ormi::core::Shape{4}, dtype
            );
            const auto mapped = ormi::core::ops::add(
                ormi::core::ops::mul(input, input), input
            );
            const auto matrix = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(), ormi::core::Shape{4, 1024}, dtype
            );
            const auto tiled = ormi::core::ops::sum(
                ormi::core::ops::add(ormi::core::ops::mul(matrix, matrix), matrix), {0}
            );
            const std::array roots{mapped, ormi::core::ops::sum(mapped), tiled};
            for (std::size_t layout = 0; layout < roots.size(); ++layout) {
                SCOPED_TRACE(static_cast<int>(dtype));
                SCOPED_TRACE(layout);
                const auto expression = ormi::core::runtime::fusion::build(*roots[layout]);
                ASSERT_TRUE(expression.has_value());
                ASSERT_EQ(ormi::backends::cuda::fusion::tile(*expression).has_value(), layout == 2);
                const auto source = ormi::backends::cuda::fusion::emit(*expression);

                for (const auto* marker : {
                    "__device__ __forceinline__ unsigned short ormi_half_bits(float value)",
                    "__device__ __forceinline__ float ormi_half_value(unsigned short value)",
                    "__device__ __forceinline__ float ormi_half_round(float value)"
                }) {
                    const auto first = source.find(marker);
                    EXPECT_EQ(first != std::string::npos, dtype == ormi::core::DType::float16);
                    if (first != std::string::npos) {
                        EXPECT_EQ(source.find(marker, first + 1), std::string::npos);
                    }
                }
                const auto* marker = "__device__ T ormi_block_sum(T value)";
                const auto first = source.find(marker);
                EXPECT_EQ(first != std::string::npos, layout == 1);
                if (first != std::string::npos) {
                    EXPECT_EQ(source.find(marker, first + 1), std::string::npos);
                }
            }
        }
    }

    TEST(CUDAFusionCodegenTest, emitsHierarchicalMapSum) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float32
        );
        const auto mapped = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_NE(source.find("float total = 0.0f;"), std::string::npos);
        EXPECT_NE(source.find("total += v3;"), std::string::npos);
        EXPECT_NE(source.find("output[task] = block;"), std::string::npos);
        EXPECT_NE(source.find("extern \"C\" __global__ void ormi_finish"), std::string::npos);
        EXPECT_NE(source.find("output[output_index] = total;"), std::string::npos);
    }

    TEST(CUDAFusionCodegenTest, emitsPartialReductionCoordinates) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3, 4},
            ormi::core::DType::float32
        );
        const auto mapped = ormi::core::ops::sin(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
        const auto output = ormi::core::ops::sum(mapped, {1}, true);
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_NE(source.find("const unsigned long long tasks = 8ull * parts;"), std::string::npos);
        EXPECT_NE(
            source.find("((output_index / 4) % 2) * 12 + (output_index % 4)"),
            std::string::npos
        );
        EXPECT_NE(source.find("reduced_index < 3ull"), std::string::npos);
        EXPECT_NE(source.find("(reduced_index % 3) * 4"), std::string::npos);
        EXPECT_NE(
            source.find("partials[output_index * parts + part]"),
            std::string::npos
        );
    }

    TEST(CUDAFusionCodegenTest, emitsRankThreePermutedReductionCoordinates) {
        const auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{5, 9, 32},
            ormi::core::DType::float32
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

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_NE(
            source.find(
                "for (unsigned long long task = blockIdx.x; "
                "task < 9ull"
            ),
            std::string::npos
        );
        EXPECT_NE(
            source.find(
                "const unsigned long long index2 = outer_index"
            ),
            std::string::npos
        );
        EXPECT_NE(source.find("reduced_index < 5ull"), std::string::npos);
        EXPECT_NE(
            source.find(
                "const unsigned long long index1 = reduced_index"
            ),
            std::string::npos
        );
        EXPECT_NE(
            source.find(
                "input0[index0 + index1 * 288 + index2 * 32]"
            ),
            std::string::npos
        );
        EXPECT_NE(
            source.find("output[index0 * 9 + index2] = total"),
            std::string::npos
        );
        EXPECT_EQ(
            source.find("extern \"C\" __global__ void ormi_finish"),
            std::string::npos
        );
    }

    TEST(CUDAFusionCodegenTest, traversesDenseGlobalInputInPhysicalOrder) {
        const auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{5, 9, 32},
            ormi::core::DType::float32
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(permuted, permuted),
            permuted
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_NE(
            source.find("const float v0 = input0[reduced_index];"),
            std::string::npos
        );
    }

    TEST(CUDAFusionCodegenTest, keepsMixedGlobalInputsInLogicalOrder) {
        const auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{5, 9, 32},
            ormi::core::DType::float32
        );
        const auto contiguous = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{32, 5, 9},
            ormi::core::DType::float32
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(permuted, permuted),
            contiguous
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_EQ(
            source.find("input0[reduced_index]"),
            std::string::npos
        );
        EXPECT_EQ(
            source.find("input1[reduced_index]"),
            std::string::npos
        );
    }

    TEST(CUDAFusionTileTest, plansCoalescedRankThreeOutputs) {
        const auto base = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{5, 9, 32},
            ormi::core::DType::float32
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

        const auto result =
            ormi::backends::cuda::fusion::tile(*expression);

        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->axis, 0u);
        EXPECT_EQ(result->width, 32u);
        EXPECT_EQ(result->tasks, 9u);
        EXPECT_EQ(result->output, (std::vector<std::size_t>{0, 2}));
        EXPECT_EQ(result->reduced, (std::vector<std::size_t>{1}));
        EXPECT_EQ(
            result->inputs,
            (std::vector<std::vector<std::size_t>>{{1, 288, 32}})
        );
    }

    TEST(CUDAFusionTileTest, leavesContiguousReductionAxisUntiled) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{257, 65},
            ormi::core::DType::float32
        );
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        EXPECT_FALSE(
            ormi::backends::cuda::fusion::tile(*expression).has_value()
        );
    }

    TEST(CUDAFusionTileTest, leavesSmallCoalescedOutputUntiled) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4096, 32},
            ormi::core::DType::float32
        );
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::sum(mapped, {0});
        const auto expression =
            ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        EXPECT_FALSE(
            ormi::backends::cuda::fusion::tile(*expression).has_value()
        );
    }

    TEST(CUDAFusionCodegenTest, emitsMeanFinalization) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{2, 3},
            ormi::core::DType::float32
        );
        const auto mapped = ormi::core::ops::add(
            ormi::core::ops::mul(input, input),
            input
        );
        const auto output = ormi::core::ops::mean(mapped, {0});
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_NE(source.find("total += v2;"), std::string::npos);
        EXPECT_NE(
            source.find("output[output_index] = total / static_cast<float>(2ull);"),
            std::string::npos
        );
    }

    TEST(CUDAFusionCodegenTest, emitsTypedTwoStageReductionFinalization) {
        for (const auto dtype : {
            ormi::core::DType::float16,
            ormi::core::DType::float32,
            ormi::core::DType::float64
        }) {
            const auto input = std::make_shared<ormi::core::Tensor>(
                ormi::core::Device::cpu(), ormi::core::Shape{2, 3, 4}, dtype
            );
            const auto mapped = ormi::core::ops::add(
                ormi::core::ops::mul(input, input), input
            );
            for (const bool mean : {false, true}) {
                SCOPED_TRACE(static_cast<int>(dtype));
                SCOPED_TRACE(mean);
                const auto output = mean
                    ? ormi::core::ops::mean(mapped, {0, 2}, true)
                    : ormi::core::ops::sum(mapped, {0, 2}, true);
                const auto expression = ormi::core::runtime::fusion::build(*output);
                ASSERT_TRUE(expression.has_value());
                ASSERT_FALSE(ormi::backends::cuda::fusion::tile(*expression));

                const auto source = ormi::backends::cuda::fusion::emit(*expression);
                const std::string storage = dtype == ormi::core::DType::float16
                    ? "unsigned short"
                    : dtype == ormi::core::DType::float64 ? "double" : "float";
                const std::string scalar = dtype == ormi::core::DType::float64
                    ? "double" : "float";
                const std::string zero = dtype == ormi::core::DType::float64
                    ? "0.0" : "0.0f";
                std::string result = mean
                    ? "total / static_cast<" + scalar + ">(8ull)"
                    : "total";
                if (dtype == ormi::core::DType::float16) {
                    result = "ormi_half_bits(" + result + ")";
                }

                const auto expected =
                    "extern \"C\" __global__ void ormi_finish(\n"
                    "    " + storage + "* output,\n"
                    "    const " + scalar + "* partials,\n"
                    "    unsigned long long parts\n"
                    ") {\n"
                    "    for (unsigned long long output_index = blockIdx.x;\n"
                    "         output_index < 3ull;\n"
                    "         output_index += gridDim.x) {\n"
                    "        " + scalar + " total = " + zero + ";\n"
                    "        for (unsigned long long part = threadIdx.x; part < parts; part += blockDim.x) {\n"
                    "            total += partials[output_index * parts + part];\n"
                    "        }\n"
                    "        total = ormi_block_sum(total);\n"
                    "        if (threadIdx.x == 0) {\n"
                    "            output[output_index] = " + result + ";\n"
                    "        }\n"
                    "    }\n"
                    "}\n";
                const auto finish = source.find("extern \"C\" __global__ void ormi_finish");
                ASSERT_NE(finish, std::string::npos);
                EXPECT_EQ(source.substr(finish), expected);
                EXPECT_NE(source.find("output[task] = block;"), std::string::npos);
            }
        }
    }

    TEST(CUDAFusionCodegenTest, rejectsAuxiliaryReductionOutputs) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float32
        );
        const auto output = ormi::core::ops::sum(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
        auto expression = ormi::core::runtime::fusion::build(*output).value();
        expression.auxiliary_outputs.push_back(expression.output);

        EXPECT_THROW(
            (void)ormi::backends::cuda::fusion::emit(expression),
            ormi::OperationError
        );
    }

    TEST(CUDAFusionCodegenTest, rejectsUnsupportedReduction) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float32
        );
        const auto output = ormi::core::ops::sum(
            ormi::core::ops::add(ormi::core::ops::mul(input, input), input)
        );
        auto expression = ormi::core::runtime::fusion::build(*output).value();
        expression.reduction->op = static_cast<ormi::core::ops::ReduceOp>(255);

        EXPECT_THROW(
            (void)ormi::backends::cuda::fusion::emit(expression),
            ormi::OperationError
        );
    }

    TEST(CUDAFusionCodegenTest, emitsTypedAutocastReduction) {
        const auto input = std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
            ormi::core::DType::float16
        );
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            output = ormi::core::ops::sum(ormi::core::ops::sin(input));
        }
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cuda::fusion::emit(*expression);

        EXPECT_NE(source.find("const unsigned short* input0"), std::string::npos);
        EXPECT_NE(
            source.find(
                "const float v0 = ormi_half_value(input0[reduced_index]);"
            ),
            std::string::npos
        );
        EXPECT_NE(source.find("const float v1 = static_cast<float>(v0);"), std::string::npos);
        EXPECT_NE(source.find("const float v2 = sinf(v1);"), std::string::npos);
        EXPECT_NE(source.find("float* output"), std::string::npos);
    }

    class CUDAFusionTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            const auto status = cudaGetDeviceCount(&device_count);

            if (status != cudaSuccess) {
                GTEST_SKIP() << "CUDA runtime is unavailable: " << cudaGetErrorString(status);
            }
            if (device_count == 0) {
                GTEST_SKIP() << "No CUDA device is available.";
            }
        }

        template <typename T>
        static std::shared_ptr<ormi::core::Tensor> tensor(
            ormi::core::DType dtype,
            const std::vector<T>& values,
            ormi::core::Shape shape
        ) {
            const auto bytes = values.size() * sizeof(T);
            auto device_buffer = std::make_shared<ormi::core::memory::Buffer>(
                ormi::core::Device::from("cuda"),
                bytes
            );
            ormi::core::memory::Buffer host(ormi::core::Device::cpu(), bytes);
            std::memcpy(host.data(), values.data(), bytes);
            ormi::core::memory::copy(*device_buffer, host);
            return std::make_shared<ormi::core::Tensor>(
                std::move(device_buffer),
                std::move(shape),
                dtype
            );
        }

        template <typename T>
        static std::shared_ptr<ormi::core::Tensor> tensor(
            ormi::core::DType dtype,
            const std::vector<T>& values
        ) {
            return tensor(
                dtype,
                values,
                ormi::core::Shape{static_cast<std::int64_t>(values.size())}
            );
        }

        template <typename T>
        static std::vector<T> host_values(ormi::core::Tensor& tensor) {
            std::vector<T> values(tensor.numel());
            ormi::core::memory::Buffer host(
                ormi::core::Device::cpu(),
                values.size() * sizeof(T)
            );
            ormi::core::memory::copy(host, *tensor.buffer());
            std::memcpy(values.data(), host.data(), host.bytes());
            return values;
        }
    };

    TEST_F(CUDAFusionTest, executesFloat32ChainWithoutRealizingIntermediates) {
        const auto input = tensor<float>(
            ormi::core::DType::float32,
            {-0.5f, 0.0f, 0.5f, 1.0f}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto sum = ormi::core::ops::add(square, input);
        const auto exponential = ormi::core::ops::exp(sum);
        const auto output = ormi::core::ops::sin(exponential);
        const auto values = host_values<float>(*output);

        for (std::size_t i = 0; i < values.size(); ++i) {
            const float x = static_cast<float>(i) * 0.5f - 0.5f;
            EXPECT_NEAR(values[i], std::sin(std::exp(x * x + x)), 1.0e-6f);
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(sum->realized());
        EXPECT_FALSE(exponential->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST_F(CUDAFusionTest, executesFloat64ChainAndReusesItsKernel) {
        for (int repetition = 0; repetition < 2; ++repetition) {
            const auto input = tensor<double>(ormi::core::DType::float64, {0.25, 0.5});
            const auto square = ormi::core::ops::mul(input, input);
            const auto output = ormi::core::ops::cos(ormi::core::ops::exp(square));
            const auto values = host_values<double>(*output);

            EXPECT_NEAR(values[0], std::cos(std::exp(0.0625)), 1.0e-12);
            EXPECT_NEAR(values[1], std::cos(std::exp(0.25)), 1.0e-12);
            EXPECT_FALSE(square->realized());
        }
    }

    TEST_F(CUDAFusionTest, executesBroadcastedFloat32Chain) {
        const auto lhs = tensor<float>(
            ormi::core::DType::float32,
            {2.0f, 4.0f},
            {2, 1}
        );
        const auto rhs = tensor<float>(
            ormi::core::DType::float32,
            {1.0f, 2.0f, 3.0f},
            {3}
        );
        const auto sum = ormi::core::ops::add(lhs, rhs);
        const auto output = ormi::core::ops::sin(ormi::core::ops::mul(sum, lhs));
        const auto values = host_values<float>(*output);

        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                const float left = row == 0 ? 2.0f : 4.0f;
                const float right = static_cast<float>(column + 1);
                EXPECT_FLOAT_EQ(values[row * 3 + column], std::sin((left + right) * left));
            }
        }
        EXPECT_FALSE(sum->realized());
    }

    TEST_F(CUDAFusionTest, executesTransposedFloat32Chain) {
        const auto base = tensor<float>(
            ormi::core::DType::float32,
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            {2, 3}
        );
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto square = ormi::core::ops::mul(transposed, transposed);
        const auto output = ormi::core::ops::sin(
            ormi::core::ops::add(square, transposed)
        );
        const auto values = host_values<float>(*output);
        const float logical[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};

        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_FLOAT_EQ(
                values[i],
                std::sin(logical[i] * logical[i] + logical[i])
            );
        }
        EXPECT_FALSE(square->realized());
    }

    TEST_F(CUDAFusionTest, executesFloat16Chain) {
        using ormi::core::Float16;
        const auto input = tensor<Float16>(
            ormi::core::DType::float16,
            {Float16{1.0f}, Float16{2.0f}}
        );
        const auto doubled = ormi::core::ops::add(input, input);
        const auto output = ormi::core::ops::mul(doubled, input);
        const auto values = host_values<Float16>(*output);

        EXPECT_FLOAT_EQ(static_cast<float>(values[0]), 2.0f);
        EXPECT_FLOAT_EQ(static_cast<float>(values[1]), 8.0f);
        EXPECT_FALSE(doubled->realized());
    }

    TEST_F(CUDAFusionTest, fusesPointwiseChainIntoGlobalSum) {
        const auto input = tensor<float>(
            ormi::core::DType::float32,
            {-0.5f, 0.0f, 0.5f, 1.0f}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped);
        const auto values = host_values<float>(*output);

        EXPECT_FLOAT_EQ(values[0], 2.5f);
        EXPECT_FALSE(mapped->realized());
        EXPECT_TRUE(output->realized());
    }

    TEST_F(CUDAFusionTest, preservesFloat16MapSumRounding) {
        using ormi::core::Float16;
        const auto input = tensor<Float16>(
            ormi::core::DType::float16,
            {Float16{0.5f}, Float16{1.0f}}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped);
        const auto values = host_values<Float16>(*output);

        EXPECT_EQ(values[0].bits(), Float16{2.75f}.bits());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, finishesMapSumAcrossMultipleBlocks) {
        std::vector<float> input_values(4096, 0.5f);
        const auto input = tensor<float>(ormi::core::DType::float32, input_values);
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped);
        const auto values = host_values<float>(*output);

        EXPECT_FLOAT_EQ(values[0], 3072.0f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesFloat64MapSum) {
        const auto input = tensor<double>(ormi::core::DType::float64, {0.25, 0.5});
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped);
        const auto values = host_values<double>(*output);

        EXPECT_DOUBLE_EQ(values[0], 1.0625);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesPointwiseChainIntoPartialSum) {
        const auto input = tensor<float>(
            ormi::core::DType::float32,
            {1.0f, 2.0f, 3.0f, 4.0f},
            {2, 2}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {0});
        const auto values = host_values<float>(*output);

        EXPECT_FLOAT_EQ(values[0], 14.0f);
        EXPECT_FLOAT_EQ(values[1], 26.0f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesPointwiseChainIntoPartialMean) {
        const auto input = tensor<float>(
            ormi::core::DType::float32,
            {1.0f, 2.0f, 3.0f, 4.0f},
            {2, 2}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped, {0});
        const auto values = host_values<float>(*output);

        EXPECT_FLOAT_EQ(values[0], 7.0f);
        EXPECT_FLOAT_EQ(values[1], 13.0f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesFloat16PartialMean) {
        using ormi::core::Float16;
        const auto input = tensor<Float16>(
            ormi::core::DType::float16,
            {Float16{0.5f}, Float16{1.0f}, Float16{1.5f}, Float16{2.0f}},
            {2, 2}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped, {1});
        const auto values = host_values<Float16>(*output);

        EXPECT_EQ(values[0].bits(), Float16{1.375f}.bits());
        EXPECT_EQ(values[1].bits(), Float16{4.875f}.bits());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, finishesMeanAcrossMultipleBlocks) {
        std::vector<float> input_values(4096, 0.5f);
        const auto input = tensor<float>(ormi::core::DType::float32, input_values);
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped);
        const auto values = host_values<float>(*output);

        EXPECT_FLOAT_EQ(values[0], 0.75f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesMultiAxisSumWithKeepdims) {
        std::vector<float> input_values(12);
        for (std::size_t i = 0; i < input_values.size(); ++i) {
            input_values[i] = static_cast<float>(i + 1);
        }
        const auto input = tensor<float>(
            ormi::core::DType::float32,
            input_values,
            {2, 3, 2}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {0, 2}, true);
        const auto values = host_values<float>(*output);

        EXPECT_EQ(output->shape(), (ormi::core::Shape{1, 3, 1}));
        EXPECT_FLOAT_EQ(values[0], 136.0f);
        EXPECT_FLOAT_EQ(values[1], 232.0f);
        EXPECT_FLOAT_EQ(values[2], 360.0f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesTransposedBroadcastPartialSum) {
        const auto base = tensor<float>(
            ormi::core::DType::float32,
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            {2, 3}
        );
        const auto rhs = tensor<float>(
            ormi::core::DType::float32,
            {0.5f, 1.0f},
            {2}
        );
        const auto transposed = ormi::core::ops::transpose(base, {1, 0});
        const auto square = ormi::core::ops::mul(transposed, transposed);
        const auto mapped = ormi::core::ops::add(square, rhs);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto values = host_values<float>(*output);

        EXPECT_FLOAT_EQ(values[0], 18.5f);
        EXPECT_FLOAT_EQ(values[1], 30.5f);
        EXPECT_FLOAT_EQ(values[2], 46.5f);
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesRankThreePermutedMiddleAxisSum) {
        constexpr std::size_t first = 32;
        constexpr std::size_t middle = 5;
        constexpr std::size_t last = 9;
        std::vector<float> input_values(first * middle * last);
        for (std::size_t j = 0; j < middle; ++j) {
            for (std::size_t k = 0; k < last; ++k) {
                for (std::size_t i = 0; i < first; ++i) {
                    input_values[(j * last + k) * first + i] =
                        static_cast<float>(i) * 0.1f +
                        static_cast<float>(k) * 0.01f;
                }
            }
        }
        const auto base = tensor<float>(
            ormi::core::DType::float32,
            input_values,
            {middle, last, first}
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto square = ormi::core::ops::mul(permuted, permuted);
        const auto mapped = ormi::core::ops::add(square, permuted);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto values = host_values<float>(*output);

        for (std::size_t i = 0; i < first; ++i) {
            for (std::size_t k = 0; k < last; ++k) {
                const auto value =
                    static_cast<float>(i) * 0.1f +
                    static_cast<float>(k) * 0.01f;
                const auto expected =
                    static_cast<float>(middle) * (value * value + value);
                EXPECT_NEAR(values[i * last + k], expected, 1.0e-5f);
            }
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesRankThreePermutedGlobalSum) {
        constexpr std::size_t first = 32;
        constexpr std::size_t middle = 5;
        constexpr std::size_t last = 9;
        std::vector<float> input_values(first * middle * last);
        float expected = 0.0f;
        for (std::size_t i = 0; i < input_values.size(); ++i) {
            const auto value =
                static_cast<float>(i % 37) * 0.01f;
            input_values[i] = value;
            expected += (value * value + value) * value;
        }
        const auto base = tensor<float>(
            ormi::core::DType::float32,
            input_values,
            {middle, last, first}
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto square = ormi::core::ops::mul(permuted, permuted);
        const auto mapped = ormi::core::ops::mul(
            ormi::core::ops::add(square, permuted),
            permuted
        );
        const auto output = ormi::core::ops::sum(mapped);
        const auto values = host_values<float>(*output);

        ASSERT_EQ(values.size(), 1u);
        EXPECT_NEAR(values[0], expected, 1.0e-3f);
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesFloat16PermutedGlobalMean) {
        using ormi::core::Float16;
        constexpr std::size_t rows = 17;
        constexpr std::size_t columns = 33;
        const auto base = tensor<Float16>(
            ormi::core::DType::float16,
            std::vector<Float16>(rows * columns, Float16{0.5f}),
            {rows, columns}
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {1, 0});
        const auto square = ormi::core::ops::mul(permuted, permuted);
        const auto mapped =
            ormi::core::ops::add(square, permuted);
        const auto output = ormi::core::ops::mean(mapped);
        const auto values = host_values<Float16>(*output);

        ASSERT_EQ(values.size(), 1u);
        EXPECT_EQ(values[0].bits(), Float16{0.75f}.bits());
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesRankThreeMixedLayoutGlobalSum) {
        constexpr std::size_t first = 7;
        constexpr std::size_t middle = 5;
        constexpr std::size_t last = 9;
        const auto base = tensor<float>(
            ormi::core::DType::float32,
            std::vector<float>(first * middle * last, 0.5f),
            {middle, last, first}
        );
        const auto contiguous = tensor<float>(
            ormi::core::DType::float32,
            std::vector<float>(first * middle * last, 0.25f),
            {first, middle, last}
        );
        const auto permuted =
            ormi::core::ops::transpose(base, {2, 0, 1});
        const auto square = ormi::core::ops::mul(permuted, permuted);
        const auto added = ormi::core::ops::add(square, contiguous);
        const auto mapped = ormi::core::ops::mul(added, permuted);
        const auto output = ormi::core::ops::sum(mapped);
        const auto values = host_values<float>(*output);

        ASSERT_EQ(values.size(), 1u);
        EXPECT_FLOAT_EQ(
            values[0],
            static_cast<float>(first * middle * last) * 0.25f
        );
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(added->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesTiledFloat16FirstAxisMean) {
        using ormi::core::Float16;
        constexpr std::size_t rows = 65;
        constexpr std::size_t columns = 257;
        const auto input = tensor<Float16>(
            ormi::core::DType::float16,
            std::vector<Float16>(rows * columns, Float16{0.5f}),
            {rows, columns}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::mean(mapped, {0});
        const auto values = host_values<Float16>(*output);

        ASSERT_EQ(values.size(), columns);
        for (const auto value : values) {
            EXPECT_EQ(value.bits(), Float16{0.75f}.bits());
        }
        EXPECT_FALSE(square->realized());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesFloat16PartialSum) {
        using ormi::core::Float16;
        const auto input = tensor<Float16>(
            ormi::core::DType::float16,
            {Float16{0.5f}, Float16{1.0f}, Float16{1.5f}, Float16{2.0f}},
            {2, 2}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto values = host_values<Float16>(*output);

        EXPECT_EQ(values[0].bits(), Float16{2.75f}.bits());
        EXPECT_EQ(values[1].bits(), Float16{9.75f}.bits());
        EXPECT_FALSE(mapped->realized());
    }

    TEST_F(CUDAFusionTest, fusesAutocastIntoFloat32Sum) {
        using ormi::core::Float16;
        const auto input = tensor<Float16>(
            ormi::core::DType::float16,
            {Float16{0.5f}, Float16{1.0f}}
        );
        std::shared_ptr<ormi::core::Tensor> mapped;
        std::shared_ptr<ormi::core::Tensor> output;
        {
            const ormi::core::Autocast autocast;
            mapped = ormi::core::ops::sin(input);
            output = ormi::core::ops::sum(mapped);
        }
        const auto values = host_values<float>(*output);

        EXPECT_FLOAT_EQ(values[0], std::sin(0.5f) + std::sin(1.0f));
        EXPECT_EQ(output->dtype(), ormi::core::DType::float32);
        EXPECT_FALSE(mapped->realized());
        EXPECT_FALSE(mapped->parents()[0]->realized());
    }

    TEST_F(CUDAFusionTest, roundsExplicitFloat16CastInsideFusion) {
        using ormi::core::Float16;
        const std::vector<float> input_values{0.3333f, 0.6667f};
        const auto input = tensor<float>(ormi::core::DType::float32, input_values);
        const auto narrowed = ormi::core::ops::cast(input, ormi::core::DType::float16);
        const auto output = ormi::core::ops::sin(narrowed);
        const auto values = host_values<Float16>(*output);

        EXPECT_EQ(
            values[0].bits(),
            Float16{std::sin(static_cast<float>(Float16{input_values[0]}))}.bits()
        );
        EXPECT_EQ(
            values[1].bits(),
            Float16{std::sin(static_cast<float>(Float16{input_values[1]}))}.bits()
        );
        EXPECT_FALSE(narrowed->realized());
    }

    TEST_F(CUDAFusionTest, fusesFloat64ToFloat32Cast) {
        const std::vector<double> input_values{0.333333333333, 0.666666666667};
        const auto input = tensor<double>(ormi::core::DType::float64, input_values);
        const auto narrowed = ormi::core::ops::cast(input, ormi::core::DType::float32);
        const auto output = ormi::core::ops::sin(narrowed);
        const auto values = host_values<float>(*output);

        EXPECT_FLOAT_EQ(values[0], std::sin(static_cast<float>(input_values[0])));
        EXPECT_FLOAT_EQ(values[1], std::sin(static_cast<float>(input_values[1])));
        EXPECT_FALSE(narrowed->realized());
    }

    TEST_F(CUDAFusionTest, finishesPartialSumsAcrossMultipleBlocks) {
        std::vector<float> input_values(8192, 0.5f);
        const auto input = tensor<float>(
            ormi::core::DType::float32,
            input_values,
            {2, 4096}
        );
        const auto square = ormi::core::ops::mul(input, input);
        const auto mapped = ormi::core::ops::add(square, input);
        const auto output = ormi::core::ops::sum(mapped, {1});
        const auto values = host_values<float>(*output);

        ASSERT_EQ(values.size(), 2u);
        EXPECT_FLOAT_EQ(values[0], 3072.0f);
        EXPECT_FLOAT_EQ(values[1], 3072.0f);
        EXPECT_FALSE(mapped->realized());
    }

}
