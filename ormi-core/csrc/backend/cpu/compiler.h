#pragma once
#include <vector>
#include <memory>
#include "core/node.h"

namespace ormi::cpu {
    struct Instruction {
        ops::OpEnum op;
        int dest;
        int src1;
        int src2;
    };

    struct CompilationResult {
        std::vector<Instruction> tape;
        std::vector<std::shared_ptr<Node>> leaf_nodes;
        std::vector<int> leaf_regs;
        int root_reg;
        int reg_count;
        int max_elements;
        std::vector<int> out_shape;
    };

    CompilationResult compile(std::shared_ptr<Node> root);
}