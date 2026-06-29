#include <gtest/gtest.h>
#include <vector>
#include <string>

#include "core/ops.h"
#include "core/error.h"
#include "backend/cuda/lowering_registry.h"

using namespace ormi;
using namespace ormi::cuda;

class CUDALoweringRegistryTest : public ::testing::Test {
protected:
    std::vector<std::string> binary_args = {"reg_1", "reg_2"};
    std::vector<std::string> unary_args = {"reg_1"};
};

TEST_F(CUDALoweringRegistryTest, BinaryOperationsSyntax) {
    auto add_fn = CUDALoweringRegistry::get(ops::OpEnum::ADD);
    EXPECT_EQ(add_fn(binary_args), "reg_1 + reg_2;");

    auto sub_fn = CUDALoweringRegistry::get(ops::OpEnum::SUB);
    EXPECT_EQ(sub_fn(binary_args), "reg_1 - reg_2;");

    auto mul_fn = CUDALoweringRegistry::get(ops::OpEnum::MUL);
    EXPECT_EQ(mul_fn(binary_args), "reg_1 * reg_2;");

    auto div_fn = CUDALoweringRegistry::get(ops::OpEnum::DIV);
    EXPECT_EQ(div_fn(binary_args), "reg_1 / reg_2;");

    auto pow_fn = CUDALoweringRegistry::get(ops::OpEnum::POW);
    EXPECT_EQ(pow_fn(binary_args), "powf(reg_1, reg_2);");
}

TEST_F(CUDALoweringRegistryTest, UnaryOperationsSyntax) {
    auto neg_fn = CUDALoweringRegistry::get(ops::OpEnum::NEG);
    EXPECT_EQ(neg_fn(unary_args), "-reg_1;");

    auto exp_fn = CUDALoweringRegistry::get(ops::OpEnum::EXP);
    EXPECT_EQ(exp_fn(unary_args), "expf(reg_1);");

    auto log_fn = CUDALoweringRegistry::get(ops::OpEnum::LOG);
    EXPECT_EQ(log_fn(unary_args), "logf(reg_1);");

    auto sin_fn = CUDALoweringRegistry::get(ops::OpEnum::SIN);
    EXPECT_EQ(sin_fn(unary_args), "sinf(reg_1);");

    auto cos_fn = CUDALoweringRegistry::get(ops::OpEnum::COS);
    EXPECT_EQ(cos_fn(unary_args), "cosf(reg_1);");
}

TEST_F(CUDALoweringRegistryTest, UnregisteredOperationThrows) {
    ops::OpEnum invalid_op = static_cast<ops::OpEnum>(9999);
    
    EXPECT_THROW(
        CUDALoweringRegistry::get(invalid_op), 
        ormi::RegistryError
    );
}