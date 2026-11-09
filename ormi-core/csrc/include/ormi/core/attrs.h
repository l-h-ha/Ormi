#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <ormi/core/float16.h>
#include <ormi/core/shape.h>

namespace ormi::core {
    namespace memory {
        class Buffer;
    }

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

    struct Pick {
        std::int64_t axis;
        Shape shape;
        std::optional<std::int32_t> ignore_index;
    };

    struct Select {
        std::size_t index;
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

    // Probability of dropping each element.
    // counter owns a separate uint64 execution counter on the output device.
    struct Bernoulli {
        double probability;
        std::uint64_t seed;
        std::shared_ptr<memory::Buffer> counter;
    };

    using Attrs = std::variant<std::monostate, Reduce, Permute, SGD, Pick, Caputo, Scalar, Bernoulli, Select>;
}
