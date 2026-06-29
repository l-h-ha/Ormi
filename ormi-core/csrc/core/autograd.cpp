#include <unordered_set>
#include <vector>
#include <stdexcept>

#include "core/autograd.h"
#include "core/registry.h"
#include "core/buffer.h"
#include "core/allocator.h"
#include "core/node_allocator.h"
#include "core/node.h"

namespace ormi {
    // Topo sort
    void build_topo(
        const std::shared_ptr<Node>& node,
        std::unordered_set<Node*>& visited,
        std::vector<std::shared_ptr<Node>>& topo
    ) {
        if (!node || visited.count(node.get())) {
            return;
        }

        visited.insert(node.get());

        for (auto& parent : node->parents) {
            build_topo(parent, visited, topo);
        }

        topo.push_back(node);
    }

    std::vector<std::shared_ptr<Node>> grad(
        const std::shared_ptr<Node>& root,
        const std::vector<std::shared_ptr<Node>>& inputs,
        const std::shared_ptr<Node>& seed
    ) {
        std::vector<std::shared_ptr<Node>> topo;
        std::unordered_set<Node*> visited;

        build_topo(root, visited, topo);

        GradContext ctx;
        ctx.set(root, seed);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const auto& node = *it;

            if (node->op == ops::OpEnum::LEAF) {
                continue;
            }

            if (!ctx.get(node)) {
                continue;
            }

            auto adjoint_func = Registry<AdjointFunc>::get(node->op);

            if (!adjoint_func) {
                ORMI_THROW(RegistryError, "Unrecognized adjoint registry.");
            }

            adjoint_func(node, ctx);
        }

        std::vector<std::shared_ptr<Node>> result;
        result.reserve(inputs.size());

        for (const auto& input : inputs) {
            result.push_back(ctx.get(input));
        }

        return result;
    }

    std::vector<std::shared_ptr<Node>> grad(const std::shared_ptr<Node>& root, const std::vector<std::shared_ptr<Node>>& inputs) {
        auto seed = ones_like(root, root->dtype, false);
        return grad(root, inputs, seed);
    }

    std::shared_ptr<Node> diff(const std::shared_ptr<Node>& y, const std::shared_ptr<Node>& x, int order) {
        if (order < 0) {
            ORMI_THROW(
                AutogradError,
                "Received a differention order of "
                + std::to_string(order)
                + ". Order must be non-negative."
            );
        }

        if (order == 0) {
            return y;
        }

        auto result = y;

        for (int i = 0; i < order; ++i) {
            result = grad(result, {x})[0];
            if (!result) {
                ORMI_THROW(AutogradError, "diff produced null gradient.");
            }
        }

        return result;
    }
}