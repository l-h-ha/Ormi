#pragma once
#include <cstdint>
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

    using Attrs = std::variant<std::monostate, Reduce, Permute, SGD, Scalar>;
}
