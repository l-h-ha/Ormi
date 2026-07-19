#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

namespace ormi::core::runtime::fusion {
    using Index = std::uint32_t;

    struct Input {
        std::size_t index;

        bool operator==(const Input&) const = default;
    };

    enum class Constant {
        zero,
        one
    };

    struct Unary {
        ops::UnaryOp op;
        Index input;

        bool operator==(const Unary&) const = default;
    };

    struct Binary {
        ops::BinaryOp op;
        Index lhs;
        Index rhs;

        bool operator==(const Binary&) const = default;
    };

    using Node = std::variant<Input, Constant, Unary, Binary>;

    struct Expr {
        DType dtype;
        std::vector<Node> nodes;
        std::vector<Tensor*> inputs;
        Index output;
        std::size_t operations;
    };

    std::optional<Expr> build(Tensor& root);
}
