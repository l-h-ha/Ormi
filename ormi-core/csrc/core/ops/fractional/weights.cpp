#include <cmath>
#include <cstddef>
#include <vector>

#include <ormi/core/error.h>

#include "core/ops/fractional/weights.h"

namespace ormi::core::ops::fractional {
    std::vector<double> weights(
        std::size_t count,
        double alpha,
        double spacing
    ) {
        if (count < 2) {
            return {};
        }

        const auto power = alpha > 1.0 ? 2.0 - alpha : 1.0 - alpha;
        const auto gamma = alpha > 1.0 ? 3.0 - alpha : 2.0 - alpha;
        const auto scale =
            std::pow(spacing, -alpha) / std::tgamma(gamma);

        if (!std::isfinite(scale)) {
            ORMI_THROW(
                OperationError,
                "Caputo scale is not representable."
            );
        }

        std::vector<double> result(count - 1);
        result[0] = scale;

        for (std::size_t k = 1; k < result.size(); ++k) {
            const auto value = static_cast<double>(k);
            const auto increment =
                std::pow(value, power) *
                std::expm1(power * std::log1p(1.0 / value));
            result[k] = scale * increment;

            if (!std::isfinite(result[k])) {
                ORMI_THROW(
                    OperationError,
                    "Caputo weights are not representable."
                );
            }
        }
        return result;
    }

    std::vector<double> adjoint_weights(
        const std::vector<double>& weights,
        double alpha
    ) {
        if (weights.empty()) {
            return {};
        }

        std::vector<double> result(weights.size());
        result[0] = weights[0];

        if (alpha < 1.0) {
            for (std::size_t k = 1; k < weights.size(); ++k) {
                result[k] = weights[k] - weights[k - 1];
            }
            return result;
        }

        if (weights.size() > 1) {
            result[1] = weights[1] - 2.0 * weights[0];
        }
        for (std::size_t k = 2; k < weights.size(); ++k) {
            result[k] =
                weights[k] - 2.0 * weights[k - 1] + weights[k - 2];
        }
        return result;
    }
}
