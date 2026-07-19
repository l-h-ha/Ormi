#pragma once
#include <cstddef>
#include <memory>
#include <string_view>
#include <variant>

#include <ormi/core/ops/kind.h>

namespace ormi::core {
    class Grads;
    class Tensor;

    enum class Output {
        allocate,
        alias
    };

    enum class Precision {
        preserve,
        lower,
        float32
    };

    using Compute = void (*)(Tensor&);
    using Adjoint = void (*)(
        const std::shared_ptr<Tensor>&,
        const std::shared_ptr<Tensor>&,
        Grads&
    );

    using ScalarOp = std::variant<std::monostate, ops::UnaryOp, ops::BinaryOp>;

    struct Op {
        std::string_view name;
        Output output;
        std::size_t parents;
        Compute compute;
        Adjoint adjoint;
        ScalarOp scalar;
        Precision precision = Precision::preserve;
    };
}
