#pragma once

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <ormi/backend/fusion.h>

#include "backends/cuda/fusion/tile.h"

namespace ormi::backends::cuda::fusion {
    inline std::string_view value_type(core::DType dtype) {
        return dtype == core::DType::float64 ? "double" : "float";
    }

    inline std::string_view literal(
        core::DType dtype,
        core::runtime::fusion::Constant constant
    ) {
        if (dtype != core::DType::float64) {
            return constant == core::runtime::fusion::Constant::zero ? "0.0f" : "1.0f";
        }

        return constant == core::runtime::fusion::Constant::zero ? "0.0" : "1.0";
    }

    inline std::string value(core::runtime::fusion::Index index) {
        return "v" + std::to_string(index);
    }

    inline std::string output_value(core::DType dtype, std::string_view expression) {
        if (dtype == core::DType::float16) {
            return "ormi_half_bits(" + std::string(expression) + ")";
        }
        return std::string(expression);
    }

    std::string offset(
        std::string_view index,
        const std::vector<core::runtime::fusion::Axis>& axes
    );

    void emit_nodes(
        std::ostringstream& source,
        const core::runtime::fusion::Expr& expression,
        const Tile* tile = nullptr,
        bool physical_order = false
    );

    void emit_reduction(
        std::ostringstream& source,
        const core::runtime::fusion::Expr& expression,
        std::string_view storage,
        std::string_view scalar,
        bool physical
    );

    void emit_tiled_reduction(
        std::ostringstream& source,
        const core::runtime::fusion::Expr& expression,
        const Tile& tile
    );
}
