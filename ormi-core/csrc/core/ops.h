#pragma once

namespace ormi::ops {
    enum class OpEnum {
        LEAF, CONST,
        NEG, EXP, LOG, SIN, COS,
        ADD, SUB, MUL, DIV, POW, MATMUL,
        SUM, TRANSPOSE, RESHAPE, EXPAND,
        FRAC_DERIV 
    };
}