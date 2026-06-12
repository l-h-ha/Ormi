#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include "compiler.cuh"
#include "lowering_registry.h"

namespace ormi::cuda {
    struct CompilerState {
        std::unordered_map<Node*, std::string> visited;
        std::vector<std::shared_ptr<Node>> leaf_nodes;
        std::vector<std::string> leaf_names;
        
        struct IRNode { std::string var; ops::OpEnum op; std::vector<std::string> operands; };
        std::vector<IRNode> tape;
        
        int var_counter = 0;
        int leaf_counter = 0;
        int tensor_size = -1;
        std::vector<int> out_shape = {1};
    };

    std::string build_ir(std::shared_ptr<Node> node, CompilerState& state) {
        if (state.visited.count(node.get())) return state.visited[node.get()];

        // Register buffer
        if (node->is_realized()) {
            std::string param_name = "LEAF_" + std::to_string(state.leaf_counter++);
            state.leaf_nodes.push_back(node);
            state.leaf_names.push_back(param_name);
            
            std::string var_name = "val_" + param_name;
            state.visited[node.get()] = var_name;
            return var_name;
        }

        // Internal node
        std::vector<std::string> operand_vars;
        for (auto& parent : node->parents) {
            operand_vars.push_back(build_ir(parent, state));
        }

        if (node->op == ops::OpEnum::MATMUL) {
            throw std::runtime_error("FATAL: MATMUL cannot be fused into a 1D element-wise kernel.");
        }

        std::string var_name = "v" + std::to_string(state.var_counter++);
        state.tape.push_back({var_name, node->op, operand_vars});
        state.visited[node.get()] = var_name;
        
        return var_name;
    }

    CompilationResult compile(std::shared_ptr<Node> root) {
        CompilerState state;
        state.out_shape = root->view.shape;
        state.tensor_size = 1;
        for (int dim : root->view.shape) {
            state.tensor_size *= dim;
        }

        std::string root_var = build_ir(root, state);

        std::stringstream src;
        src << "extern \"C\" __global__\n";
        src << "void fused_kernel(float* OUT_Z";
        for (const auto& name : state.leaf_names) {
            src << ", const float* " << name;
        }
        src << ") {\n";
        src << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        src << "    if (idx >= " << state.tensor_size << ") return;\n\n";

        int ndim = state.out_shape.size();
        std::vector<int> out_strides = root->view.strides;

        for (size_t i = 0; i < state.leaf_names.size(); ++i) {
            const auto& leaf_view = state.leaf_nodes[i]->view;
            bool is_scalar = leaf_view.shape.empty() || (state.leaf_nodes[i]->data->bytes / sizeof(float) == 1);
            
            if (is_scalar) {
                src << "    float val_" << state.leaf_names[i] << " = " << state.leaf_names[i] << "[0];\n";
            } else {
                int offset = ndim - leaf_view.shape.size();

                std::vector<int> leaf_strides(ndim, 0);
                for (int d = 0; d < ndim; ++d) {
                    int orig_d = d - offset;
                    if (orig_d >= 0 && leaf_view.shape[orig_d] != 1) {
                        leaf_strides[d] = leaf_view.strides[orig_d];
                    }
                }
                
                src << "    int leaf_idx_" << i << " = 0;\n";
                src << "    int remaining_" << i << " = idx;\n";
                for (int d = 0; d < ndim; ++d) {
                    src << "    leaf_idx_" << i << " += (remaining_" << i << " / " << out_strides[d] << ") * " << leaf_strides[d] << ";\n";
                    src << "    remaining_" << i << " %= " << out_strides[d] << ";\n";
                }
                src << "    float val_" << state.leaf_names[i] << " = " << state.leaf_names[i] << "[leaf_idx_" << i << "];\n";
            }
        }
        src << "\n";
        
        for (const auto& node : state.tape) {
            src << "    float " << node.var << " = ";
            auto lowering_func = CUDALoweringRegistry::get(node.op);
            src << lowering_func(node.operands) << "\n";
        }

        src << "\n    OUT_Z[idx] = " << root_var << ";\n}\n";
        return {src.str(), state.leaf_nodes, state.leaf_names, state.tensor_size, state.out_shape};
    }
}