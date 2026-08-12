#pragma once
#include <unordered_set>
#include <vector>

#include "core/runtime/fusion/group.h"
#include "core/runtime/step.h"

namespace ormi::core::runtime::schedule {
    void run(Tensor& root);
    void append(
        Tensor& root,
        std::unordered_set<const Tensor*>& completed,
        std::vector<Step>& steps,
        const fusion::Groups& groups = {}
    );
}
