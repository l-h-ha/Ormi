#include <unordered_map>
#include <stdexcept>
#include "compiler.h"
#include "lowering_registry.h"
#include "core/ops.h"

namespace ormi::cpu {
    static bool is_jit_lowerable(ops::OpEnum op) {
        if (op == ops::OpEnum::LEAF) {
            return false;
        }

        try {
            (void)CPULoweringRegistry::get(op);
            return true;
        } catch (...) {
            return false;
        }
    }

    struct VMState {
        std::unordered_map<Node*, int> node_to_reg;
        std::vector<std::shared_ptr<Node>> leaf_nodes;
        std::vector<int> leaf_regs;
        std::vector<Instruction> tape;
        
        int reg_counter = 0;
        int max_elements = 1;
        std::vector<int> out_shape;
    };

    int build_bytecode(const std::shared_ptr<Node>& node, VMState& state) {
        if (state.node_to_reg.count(node.get())) return state.node_to_reg[node.get()];

        int reg = state.reg_counter++;
        state.node_to_reg[node.get()] = reg;

        // Stop only at true leaves or non-JIT ops.
        // Do not stop at realized pointwise/JIT nodes, because ComputeTape
        // must recompute them when parameter buffers change.
        if (node->op == ops::OpEnum::LEAF || !is_jit_lowerable(node->op)) {
            state.leaf_nodes.push_back(node);
            state.leaf_regs.push_back(reg);
            return reg;
        }

        // Parse parents of internal nodes
        int s1 = -1, s2 = -1;
        if (node->parents.size() > 0) s1 = build_bytecode(node->parents[0], state);
        if (node->parents.size() > 1) s2 = build_bytecode(node->parents[1], state);

        // Write instruction to tape
        state.tape.push_back({node->op, reg, s1, s2});
        return reg;
    }

    CompilationResult compile(const std::shared_ptr<Node>& root) {
        VMState state;
        
        state.out_shape = root->view.shape;
        state.max_elements = 1;
        for (int dim : root->view.shape) {
            state.max_elements *= dim;
        }

        int root_reg = build_bytecode(root, state);
        
        return {
            state.tape, state.leaf_nodes, state.leaf_regs, 
            root_reg, state.reg_counter, state.max_elements, state.out_shape
        };
    }
}