#pragma once
#include <span>
#include <unordered_map>
#include <vector>

#include <ormi/core/tensor.h>

namespace ormi::core::runtime::fusion {
    using Candidates = std::unordered_map<const Tensor*, std::vector<Tensor*>>;

    Candidates find_candidates(std::span<Tensor* const> roots);
}
