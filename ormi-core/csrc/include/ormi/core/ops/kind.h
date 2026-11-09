#pragma once

namespace ormi::core::ops {
    enum class BinaryOp { add, sub, mul, div, minimum, maximum };

    enum class ReduceOp { sum, mean, max };

    enum class FractionalOp { caputo, caputo_adjoint, caputo_initial, caputo_initial_adjoint };

    enum class UnaryOp {
        neg,
        exp,
        log,
        sin,
        cos,
        abs,
        sign,
        positive,
        tanh,
        sigmoid,
        softplus,
        relu,
        erf,
        log1p,
        expm1,
        rsqrt
    };
}
