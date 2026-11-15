#pragma once
#include <cstddef>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ormi/core/tensor.h>

namespace ormi::core::runtime::fusion {
    using Candidates = std::unordered_map<const Tensor*, std::vector<Tensor*>>;

    Candidates find_candidates(
        std::span<Tensor* const> roots,
        std::unordered_set<const Tensor*>* shared_scalars = nullptr,
        std::size_t minimum_nodes = 0
    );
}
