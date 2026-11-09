#pragma once
#include <unordered_set>

#include <ormi/core/tensor.h>

#include "core/runtime/fusion/group.h"
#include "core/runtime/step.h"

namespace ormi::core::runtime {
    enum class Preparation {
        direct,
        deferred
    };

    Step lower(Tensor& tensor, Preparation preparation);
    Step lower(Tensor& tensor, const fusion::Candidates& candidates, Preparation preparation);
    Step lower(
        Tensor& tensor,
        const fusion::Candidates& candidates,
        const std::unordered_set<const Tensor*>& boundaries,
        Preparation preparation
    );
}
