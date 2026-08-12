#pragma once
#include <span>
#include <unordered_map>
#include <vector>

#include <ormi/core/tensor.h>

namespace ormi::core::runtime::fusion {
    using Groups = std::unordered_map<const Tensor*, std::vector<Tensor*>>;

    Groups groups(std::span<Tensor* const> roots);
}
