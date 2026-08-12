#pragma once
#include <cstddef>
#include <vector>

namespace ormi::core::ops::fractional {
    std::vector<double> weights(
        std::size_t count,
        double alpha,
        double spacing
    );

    std::vector<double> adjoint_weights(
        const std::vector<double>& weights,
        double alpha
    );
}
