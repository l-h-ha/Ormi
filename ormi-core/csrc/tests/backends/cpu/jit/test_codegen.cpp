#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/jit/codegen.h"
#include "core/runtime/fusion/expr.h"

namespace {
    std::shared_ptr<ormi::core::Tensor> tensor(ormi::core::DType dtype) {
        return std::make_shared<ormi::core::Tensor>(
            ormi::core::Device::cpu(),
            ormi::core::Shape{4},
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
        EXPECT_NE(first.find("auto* output = static_cast<float*>(output_ptr)"), std::string::npos);
        EXPECT_NE(first.find("const float v1 = v0 * v0;"), std::string::npos);
        EXPECT_NE(first.find("const float v2 = v1 + v0;"), std::string::npos);
        EXPECT_NE(first.find("const float v3 = std::sin(v2);"), std::string::npos);
        EXPECT_NE(first.find("output[i] = v3;"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, emitsFloat64Kernel) {
        const auto x = tensor(ormi::core::DType::float64);
        const auto output = ormi::core::ops::cos(ormi::core::ops::exp(x));
        const auto expression = ormi::core::runtime::fusion::build(*output);
        ASSERT_TRUE(expression.has_value());

        const auto source = ormi::backends::cpu::jit::emit(*expression);

        EXPECT_NE(source.find("auto* output = static_cast<double*>(output_ptr)"), std::string::npos);
        EXPECT_NE(source.find("const double v1 = std::exp(v0);"), std::string::npos);
        EXPECT_NE(source.find("const double v2 = std::cos(v1);"), std::string::npos);
    }

    TEST(CPUJITCodegenTest, rejectsMalformedExpression) {
        ormi::core::runtime::fusion::Expr expression{
            ormi::core::DType::float32,
            {},
            {},
            0,
            0
        };

        EXPECT_THROW((void)ormi::backends::cpu::jit::emit(expression), ormi::OperationError);
    }
}
