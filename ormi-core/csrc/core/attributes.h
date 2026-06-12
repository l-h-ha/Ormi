#pragma once
#include <vector>
#include <variant>

namespace ormi {
    struct NoAttrs {};

    struct ReduceAttrs {
        std::vector<int> axes;
        bool keepdims = false;
    };

    struct PermuteAttrs {
        std::vector<int> axes;
    };

    struct ReshapeAttrs {
        std::vector<int> shape;
    };

    struct ExpandAttrs {
        std::vector<int> shape;
    };

    using OpAttributes = std::variant<
        NoAttrs,
        ReduceAttrs,
        PermuteAttrs,
        ReshapeAttrs,
        ExpandAttrs
    >;
}