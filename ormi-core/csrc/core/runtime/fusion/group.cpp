#include <algorithm>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ormi/core/tensor.h>

#include "core/runtime/fusion/group.h"

namespace ormi::core::runtime::fusion {
    namespace {
        struct Primaries {
            Tensor* first;
            std::vector<Tensor*> rest;
        };

        bool same_output(const Tensor& lhs, const Tensor& rhs) {
            return
                lhs.dtype() == rhs.dtype() &&
                lhs.shape() == rhs.shape() &&
                lhs.device().backend_id() == rhs.device().backend_id() &&
                lhs.device().index() == rhs.device().index();
        }

        Tensor* match(const Primaries& primaries, const Tensor& output) {
            if (same_output(*primaries.first, output)) {
                return primaries.first;
            }

            const auto primary = std::find_if(
                primaries.rest.begin(),
                primaries.rest.end(),
                [&output](const Tensor* candidate) {
                    return same_output(*candidate, output);
                }
            );
            return primary == primaries.rest.end() ? nullptr : *primary;
        }
    }

    Candidates find_candidates(std::span<Tensor* const> roots) {
        std::vector<Tensor*> stack;
        stack.reserve(roots.size());
        for (auto root = roots.rbegin(); root != roots.rend(); ++root) {
            if (*root) {
                stack.push_back(*root);
            }
        }

        std::unordered_set<const Tensor*> visited;
        std::unordered_map<const Tensor*, Primaries> primaries;
        Candidates result;

        while (!stack.empty()) {
            auto* tensor = stack.back();
            stack.pop_back();
            if (tensor->realized()) {
                continue;
            }
            if (!visited.insert(tensor).second) {
                continue;
            }

            if (tensor->parents().size() == 1) {
                const auto parent = tensor->parents()[0].get();
                const auto [entry, inserted] = primaries.try_emplace(
                    parent,
                    Primaries{tensor, {}}
                );
                if (!inserted) {
                    if (auto* primary = match(entry->second, *tensor)) {
                        result[primary].push_back(tensor);
                    } else {
                        entry->second.rest.push_back(tensor);
                    }
                }
            }

            for (auto parent = tensor->parents().rbegin();
                 parent != tensor->parents().rend();
                 ++parent) {
                stack.push_back(parent->get());
            }
        }

        return result;
    }
}
