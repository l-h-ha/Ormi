#include <unordered_set>
#include <vector>
#include <stdexcept>

#include "core/autograd.h"
#include "core/registry.h"
#include "core/memory.h"
#include "core/allocator.h"

namespace ormi {
    // Topo sort
    void build_topo(std::shared_ptr<Node> node, 
                    std::unordered_set<Node*>& visited, 
                    std::vector<std::shared_ptr<Node>>& topo) {
        if (!node || visited.count(node.get())) return;
        visited.insert(node.get());
        
        for (auto& parent : node->parents) {
            build_topo(parent, visited, topo);
        }
        topo.push_back(node);
    }
    
    void compute_gradients(std::shared_ptr<Node> root) {
        std::vector<std::shared_ptr<Node>> topo;
        std::unordered_set<Node*> visited;
        build_topo(root, visited, topo);
    
        // Seed
        if (!root->grad) {
            auto seed_buf = ones(root->view.shape, root->device.type);
            root->grad = std::make_shared<Node>(seed_buf, false, root->device);
        }
    
        // Backprop
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            auto node = *it;
            if (!node->grad || node->op == ops::OpEnum::LEAF) continue;

            auto adjoint_func = Registry<AdjointFunc>::get(node->op);
            if (adjoint_func) {
                adjoint_func(node);
            } else {
                throw std::runtime_error("No Adjoint rule registered for operation.");
            }
        }
    }
    
    void zero_grad_graph(std::shared_ptr<Node> root) {
        std::vector<std::shared_ptr<Node>> topo;
        std::unordered_set<Node*> visited;
        
        build_topo(root, visited, topo);
        
        for (auto& node : topo) {
            node->grad = nullptr;
        }
    }
}