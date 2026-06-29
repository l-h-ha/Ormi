#pragma once
#include <string>
#include <stdexcept>
#include <cctype>
#include <algorithm>
#include <magic_enum.hpp>
#include "core/error.h"

namespace ormi::ops {
    enum class OpEnum {
        LEAF, CONST,
        NEG, EXP, LOG, SIN, COS,
        ADD, SUB, MUL, DIV, POW, MATMUL,
        SUM, TRANSPOSE, RESHAPE, EXPAND,
        FRAC_DERIV
    };

    inline std::string to_string(OpEnum op) {
        auto name = magic_enum::enum_name(op);
        if (name.empty()) {
            ORMI_THROW(DTypeError, "Unrecognized DType.");
        }

        std::string t(name);
        std::transform(t.begin(), t.end(), t.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        return t;
    }
}