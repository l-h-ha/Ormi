#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <ormi/core/dtype.h>
#include <ormi/core/ops/kind.h>

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::runtime::fusion {
    using Index = std::uint32_t;

    struct Axis {
        std::size_t divisor;
        std::size_t extent;
        std::size_t stride;

        bool operator==(const Axis&) const = default;
    };

    struct Iteration {
        std::size_t size;
        std::vector<Axis> axes;

        bool operator==(const Iteration&) const = default;
    };

    struct Input {
        std::size_t index;
        std::vector<Axis> axes;
        bool linear;

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

    struct Cast {
        Index input;

        bool operator==(const Cast&) const = default;
    };

    struct Binary {
        ops::BinaryOp op;
        Index lhs;
        Index rhs;

        bool operator==(const Binary&) const = default;
    };

    using Instruction = std::variant<Input, Constant, Cast, Unary, Binary>;

    struct Node {
        DType dtype;
        Instruction instruction;

        bool operator==(const Node&) const = default;
    };

    struct Reduction {
        ops::ReduceOp op;
        Iteration output;
        Iteration reduced;

        bool operator==(const Reduction&) const = default;
    };

    struct Expr {
        DType dtype;
        std::vector<std::size_t> shape;
        std::vector<Node> nodes;
        std::vector<Tensor*> inputs;
        Index output;
        std::vector<Index> auxiliary_outputs;
        std::size_t operations;
        std::size_t elements;
        std::optional<Reduction> reduction;
    };
}
