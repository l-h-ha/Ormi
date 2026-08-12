#pragma once
#include <optional>
#include <span>
#include <unordered_set>

#include <ormi/core/tensor.h>

#include "core/runtime/step.h"

namespace ormi::core::runtime {
    Step lower(Tensor& tensor);
    Step lower(
        Tensor& tensor,
        const std::unordered_set<const Tensor*>& boundaries
    );
    std::optional<Step> lower_group(
        Tensor& tensor,
        std::span<Tensor* const> auxiliary_outputs
    );
    std::optional<Step> lower_group(
        Tensor& tensor,
        std::span<Tensor* const> auxiliary_outputs,
        const std::unordered_set<const Tensor*>& boundaries
    );
}
