#include <stdexcept>
#include <cmath>
#include <unordered_map>

#include "jit.h"
#include "compiler.h"
#include "core/memory.h"
#include "core/shape.h"
#include "lowering_registry.h"

namespace ormi::cpu {
    void execute_jit(std::shared_ptr<Node> root) {
        CompilationResult ast = compile(root);
        root->data = empty(ast.out_shape, DeviceEnum::CPU);
        root->view.shape = ast.out_shape;
        root->view.strides = calc_contiguous_strides(ast.out_shape);

        float* out_ptr = static_cast<float*>(root->data->ptr);
        
        std::vector<float*> leaves;
        std::vector<bool> is_scalar;
        for (auto& node : ast.leaf_nodes) {
            leaves.push_back(static_cast<float*>(node->data->ptr));
            is_scalar.push_back(node->view.shape.empty() || (node->data->bytes / sizeof(float) == 1));
        }

        int ndim = ast.out_shape.size();
        std::vector<int> out_strides = root->view.strides; 
        std::vector<std::vector<int>> leaf_strides(leaves.size(), std::vector<int>(ndim, 0));
        
        for (size_t L = 0; L < leaves.size(); ++L) {
            if (is_scalar[L]) continue;

            const auto& leaf_view = ast.leaf_nodes[L]->view;
            int offset = ndim - leaf_view.shape.size();

            for (int d = 0; d < ndim; ++d) {
                int original_dim_idx = d - offset;

                if (original_dim_idx >= 0 && leaf_view.shape[original_dim_idx] != 1) {
                    leaf_strides[L][d] = leaf_view.strides[original_dim_idx];
                } else {
                    leaf_strides[L][d] = 0;
                }
            }
        }
        
        ///
        ///
        ///

        std::vector<LoweringFunc> resolved_tape;
        resolved_tape.reserve(ast.tape.size());
        for (const auto& inst : ast.tape) {
            resolved_tape.push_back(CPULoweringRegistry::get(inst.op));
        }

        std::vector<float> regs(ast.reg_count, 0.0f);
        for (int i = 0; i < ast.max_elements; ++i) {
            // Load variables
            for (size_t L = 0; L < leaves.size(); ++L) {
                if (is_scalar[L]) {
                    regs[ast.leaf_regs[L]] = leaves[L][0];
                } else {
                    int leaf_idx = 0;
                    int remaining = i;
                    for (int d = 0; d < ndim; ++d) {
                        int coord = remaining / out_strides[d];
                        remaining %= out_strides[d];
                        leaf_idx += coord * leaf_strides[L][d];
                    }
                    regs[ast.leaf_regs[L]] = leaves[L][leaf_idx];
                }
            }

            // Execute tape
            for (size_t t = 0; t < ast.tape.size(); ++t) {
                const auto& inst = ast.tape[t];
                resolved_tape[t](regs, inst.dest, inst.src1, inst.src2);
            }

            // Store result
            out_ptr[i] = regs[ast.root_reg];
        }
    }
}