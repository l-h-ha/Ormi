#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include <ormi/core/float16.h>

namespace ormi::core {
    using Scalar = std::variant<double, float, Float16, std::int32_t, bool>;

    struct Reduce {
        std::vector<std::int64_t> axes;
        bool keepdims;
    };

    struct Permute {
        std::vector<std::int64_t> axes;
    };

    struct SGD {
        double step_size;
    };

    struct Caputo {
        struct Coefficients {
            std::vector<double> forward;
            std::vector<double> adjoint;
        };

        double alpha;
        double spacing;
        std::int64_t axis;
        std::size_t count;
        std::shared_ptr<const Coefficients> coefficients;
    };

    using Attrs = std::variant<
        std::monostate,
        Reduce,
        Permute,
        SGD,
        Caputo,
        Scalar
    >;
}
