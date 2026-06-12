#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <stdexcept>

#include "core/node.h"
#include "core/memory.h"
#include "core/ops.h"
#include "backend/cpu/compiler.h"
#include "core/ops/pointwise/add.h"
#include "core/ops/pointwise/mul.h"
#include "core/ops/blas/matmul.h"

using namespace ormi;

class CPUCompilerTest : public ::testing::Test {
protected:
    std::shared_ptr<Node> make_leaf(const std::vector<int>& shape) {
        auto dev = Device(DeviceEnum::CPU);
        auto buf = empty(shape, dev.type);
        return std::make_shared<Node>(buf, false, dev);
    }
};

TEST_F(CPUCompilerTest, BasicTapeConstruction) {
    auto a = make_leaf({2, 2});
    auto b = make_leaf({2, 2});
    auto c = ops::add(a, b);

    auto result = cpu::compile(c);

    // 3 total registers (a, b, c)
    EXPECT_EQ(result.reg_count, 3);
    
    // 2 leaves
    EXPECT_EQ(result.leaf_nodes.size(), 2);
    EXPECT_EQ(result.leaf_regs.size(), 2);
    
    // 1 instruction
    ASSERT_EQ(result.tape.size(), 1);
    EXPECT_EQ(result.tape[0].op, ops::OpEnum::ADD);
    EXPECT_EQ(result.tape[0].dest, result.root_reg);
}

TEST_F(CPUCompilerTest, NodeDeduplication) {
    auto a = make_leaf({2, 2});
    auto c = ops::add(a, a); 

    auto result = cpu::compile(c);

    // 2 registers total (a, c) despite 'a' being used twice
    EXPECT_EQ(result.reg_count, 2);
    EXPECT_EQ(result.leaf_nodes.size(), 1);
    
    ASSERT_EQ(result.tape.size(), 1);
    
    // Both source registers must point to the exact same virtual register
    EXPECT_EQ(result.tape[0].src1, result.tape[0].src2);
}

TEST_F(CPUCompilerTest, ComplexDAGTraversal) {
    // D = (A + B) * A
    auto a = make_leaf({2});
    auto b = make_leaf({2});
    auto a_plus_b = ops::add(a, b);
    auto d = ops::mul(a_plus_b, a);

    auto result = cpu::compile(d);

    // 4 registers: A, B, A+B, D
    EXPECT_EQ(result.reg_count, 4);
    
    // Tape must resolve ADD before MUL
    ASSERT_EQ(result.tape.size(), 2);
    EXPECT_EQ(result.tape[0].op, ops::OpEnum::ADD);
    EXPECT_EQ(result.tape[1].op, ops::OpEnum::MUL);

    // MUL's destination must be the root register
    EXPECT_EQ(result.tape[1].dest, result.root_reg);
}

TEST_F(CPUCompilerTest, GraphBreakerValidation) {
    auto a = make_leaf({2, 2});
    auto b = make_leaf({2, 2});
    auto c = ops::matmul(a, b);

    // The compiler must refuse to trace a graph breaking operation
    EXPECT_THROW(cpu::compile(c), std::runtime_error);
}