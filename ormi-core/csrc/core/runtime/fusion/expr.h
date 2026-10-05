#pragma once
#include <optional>
#include <span>
#include <unordered_set>

#include <ormi/backend/fusion.h>
#include <ormi/core/tensor.h>

namespace ormi::core::runtime::fusion {
    std::optional<Expr> build(Tensor& root);
    std::optional<Expr> build(
        Tensor& root,
        const std::unordered_set<const Tensor*>& boundaries
    );
    std::optional<Expr> build(
        Tensor& root,
        std::span<Tensor* const> auxiliary_outputs
    );
    std::optional<Expr> build(
        Tensor& root,
        std::span<Tensor* const> auxiliary_outputs,
        const std::unordered_set<const Tensor*>& boundaries
    );
}
