#include <array>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/ops/reduction.h>
#include <ormi/core/ops/view.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cuda/fusion/codegen.h"
#include "backends/cuda/fusion/tile.h"
#include "core/runtime/fusion/expr.h"

namespace {
    namespace core = ormi::core;
    namespace ops = core::ops;
    namespace fusion = core::runtime::fusion;

    constexpr std::array dtypes{core::DType::float16, core::DType::float32, core::DType::float64};

    std::shared_ptr<core::Tensor> input(core::Shape shape, core::DType dtype) {
        return std::make_shared<core::Tensor>(core::Device::cpu(), std::move(shape), dtype);
    }

    void expect_load(const std::string& source, core::DType dtype, const std::string& access) {
        const auto expected = dtype == core::DType::float16
            ? "ormi_half_value(" + access + ")" : access;
        EXPECT_NE(source.find("= " + expected + ";"), std::string::npos);
    }

    TEST(CUDAFusionInputTest, emitsLinearAndScalarBroadcastLoads) {
        for (const auto dtype : dtypes) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const auto values = input({2, 3, 4}, dtype);
            const auto scalar = input({}, dtype);
            const auto output = ops::add(ops::mul(values, values), scalar);
            const auto expression = fusion::build(*output);
            ASSERT_TRUE(expression);
            ASSERT_EQ(expression->inputs.size(), 2u);
            ASSERT_EQ(expression->inputs[0], values.get());
            ASSERT_EQ(expression->inputs[1], scalar.get());

            const auto source = ormi::backends::cuda::fusion::emit(*expression);
            expect_load(source, dtype, "input0[i]");
            expect_load(source, dtype, "input1[0]");
        }
    }

    TEST(CUDAFusionInputTest, emitsRankFourPermutedOffsets) {
        for (const auto dtype : dtypes) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const auto values = ops::transpose(input({2, 3, 5, 7}, dtype), {2, 0, 3, 1});
            const auto output = ops::add(ops::mul(values, values), values);
            const auto expression = fusion::build(*output);
            ASSERT_TRUE(expression);
            ASSERT_EQ(expression->inputs.size(), 1u);

            const auto source = ormi::backends::cuda::fusion::emit(*expression);
            expect_load(
                source, dtype,
                "input0[((i / 42) % 5) * 7 + ((i / 21) % 2) * 105 + "
                "((i / 3) % 7) + (i % 3) * 35]"
            );
        }
    }

    TEST(CUDAFusionInputTest, emitsTiledBroadcastLoadsWithZeroAndUnitStrides) {
        for (const auto dtype : dtypes) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const auto values = input({3, 257}, dtype);
            const auto row = input({257}, dtype);
            const auto scalar = input({}, dtype);
            const auto output = ops::sum(ops::add(ops::add(values, row), scalar), {0});
            const auto expression = fusion::build(*output);
            ASSERT_TRUE(expression);
            ASSERT_EQ(expression->inputs.size(), 3u);
            ASSERT_TRUE(ormi::backends::cuda::fusion::tile(*expression));

            const auto source = ormi::backends::cuda::fusion::emit(*expression);
            expect_load(source, dtype, "input0[index0 * 257 + index1]");
            expect_load(source, dtype, "input1[index1]");
            expect_load(source, dtype, "input2[0]");
            EXPECT_EQ(source.find("input0[i]"), std::string::npos);
        }
    }

    TEST(CUDAFusionInputTest, rejectsInvalidInputBeforeFormattingItsAddress) {
        const auto values = input({3, 257}, core::DType::float32);
        const auto mapped = ops::add(ops::mul(values, values), values);
        const std::array outputs{mapped, ops::sum(mapped), ops::sum(mapped, {0})};
        for (const auto& output : outputs) {
            auto expression = fusion::build(*output);
            ASSERT_TRUE(expression);
            auto* node = std::get_if<fusion::Input>(&expression->nodes.front().instruction);
            ASSERT_NE(node, nullptr);
            node->index = expression->inputs.size();

            try {
                (void)ormi::backends::cuda::fusion::emit(*expression);
                FAIL() << "Code generation accepted an invalid input reference.";
            } catch (const ormi::OperationError& error) {
                EXPECT_STREQ(error.what(), "Fusion expression references an invalid input.");
            }
        }
    }
}
