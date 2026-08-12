#pragma once

namespace ormi::core::ops {
    enum class BinaryOp {
        add,
        sub,
        mul,
        div
    };

    enum class ReduceOp {
        sum,
        mean
    };

    enum class FractionalOp {
        caputo,
        caputo_adjoint,
        caputo_initial,
        caputo_initial_adjoint
    };

    enum class UnaryOp {
        neg,
        exp,
        log,
        sin,
        cos
    };
}
