#pragma once
#include <cstddef>
#include <memory>
#include <string_view>
#include <variant>

#include <ormi/core/ops/kind.h>
#include <ormi/core/setup.h>

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

    // Parents are realized before compute runs. Allocating outputs already have storage.
    // Alias callbacks run before the output borrows storage and must not call out.data().
    using Compute = void (*)(Tensor&);
    // Return owned kernel/workspace state for repeated execution, or an empty Setup for compute.
    using Prepare = Setup (*)(Tensor&);
    // Adjoint: build gradient expressions and add them to Grads without running kernels.
    using Adjoint = void (*)(
        const std::shared_ptr<Tensor>&,
        const std::shared_ptr<Tensor>&,
        Grads&
    );

    using ScalarOp = std::variant<std::monostate, ops::UnaryOp, ops::BinaryOp>;

    // Each operation has one static descriptor. Nodes point to it through OpRef.
    // scalar identifies operations the fusion builder can put inside a kernel.
    struct Op {
        std::string_view name;
        Output output;
        std::size_t parents;
        Compute compute;
        Prepare prepare = nullptr;
        Adjoint adjoint;
        ScalarOp scalar;
        Precision precision = Precision::preserve;
    };

    class OpRef final {
    public:
        constexpr const Op* get() const noexcept {
            return op_;
        }

    private:
        constexpr explicit OpRef(const Op* op) noexcept : op_(op) {}

        template <const Op&>
        friend consteval OpRef op_ref() noexcept;

        const Op* op_;
    };

    template <const Op& op>
    consteval OpRef op_ref() noexcept {
        return OpRef(&op);
    }

    static_assert(sizeof(OpRef) == sizeof(const Op*));
}
