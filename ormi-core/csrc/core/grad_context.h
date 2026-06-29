#pragma once

#include <memory>
#include <unordered_map>

#include "core/node.h"
#include "core/node_allocator.h"

namespace ormi {
    struct GradContext {
        std::unordered_map<Node*, std::shared_ptr<Node>> grads;

        std::shared_ptr<Node> get(const std::shared_ptr<Node>& node) const {
            auto it = grads.find(node.get());

            if (it == grads.end()) {
                return nullptr;
            }

            return it->second;
        }

        void set(const std::shared_ptr<Node>& node, const std::shared_ptr<Node>& grad) {
            grads[node.get()] = grad;
        }

        void add(const std::shared_ptr<Node>& node, const std::shared_ptr<Node>& grad) {
            auto existing = get(node);

            if (!existing) {
                set(node, grad);
                return;
            }

            auto accumulated = std::allocate_shared<Node>(
                get_node_allocator<Node>(),
                ops::OpEnum::ADD,
                std::vector<std::shared_ptr<Node>>{existing, grad}
            );

            set(node, accumulated);
        }
    };
}