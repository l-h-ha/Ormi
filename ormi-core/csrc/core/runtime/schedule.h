#pragma once
#include <unordered_set>
#include <vector>

#include "core/runtime/fusion/group.h"
#include "core/runtime/step.h"

namespace ormi::core::runtime::schedule {
    // Walk missing dependencies in parent-before-child order, then execute them.
    void run(Tensor& root);
    // Append steps without running them. completed avoids duplicate steps across Plan outputs
    // and keeps later fusion groups from taking nodes that already have a step.
    void append(
        Tensor& root,
        std::unordered_set<const Tensor*>& completed,
        std::vector<Step>& steps,
        const fusion::Candidates& candidates = {}
    );
}
