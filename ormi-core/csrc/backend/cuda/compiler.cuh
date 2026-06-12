#pragma once
#include <string>
#include <vector>
#include <memory>
#include "core/node.h"

namespace ormi::cuda {
    struct CompilationResult {
        std::string kernel_code;
        std::vector<std::shared_ptr<Node>> leaf_nodes;
        std::vector<std::string> leaf_names;
        int tensor_size;
        std::vector<int> out_shape;
    };

    CompilationResult compile(std::shared_ptr<Node> root);
}