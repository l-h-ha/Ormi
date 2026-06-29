#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include "core/node.h"

namespace ormi {
    class Tape {
    public:
        virtual ~Tape() = default;
        virtual void execute() = 0;
    };

    namespace Topology {
        void _build(
            const std::shared_ptr<Node>& node, 
            std::unordered_set<Node*>& visited, 
            std::vector<std::shared_ptr<Node>>& sorted
        );

        std::vector<std::shared_ptr<Node>> sort(const std::vector<std::shared_ptr<Node>>& roots);
    }
}