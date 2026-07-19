#pragma once

namespace ormi::core::ops {
    enum class BinaryOp {
        add,
        sub,
        mul,
        div
    };

    enum class ReduceOp {
        sum
    };

    enum class UnaryOp {
        neg,
        exp,
        log,
        sin,
        cos
    };
}
