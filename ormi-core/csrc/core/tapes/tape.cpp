#include "tape.h"

namespace ormi::Topology {
    void _build(
        const std::shared_ptr<Node>& node, 
        std::unordered_set<Node*>& visited, 
        std::vector<std::shared_ptr<Node>>& sorted
    ) {
        if (!node || visited.count(node.get())) return;
        visited.insert(node.get());
        
        for (auto& parent : node->parents) {
            _build(parent, visited, sorted);
        }
        sorted.push_back(node);
    }

    std::vector<std::shared_ptr<Node>> sort(const std::vector<std::shared_ptr<Node>>& roots) {
        std::vector<std::shared_ptr<Node>> sorted;
        std::unordered_set<Node*> visited;
        for (const auto& root : roots) {
            _build(root, visited, sorted);
        }
        return sorted;
    }
}