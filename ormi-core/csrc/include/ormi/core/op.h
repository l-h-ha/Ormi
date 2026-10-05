#pragma once
#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>
#include <variant>

#include <ormi/core/ops/kind.h>

namespace ormi::core {
    class Grads;
    class Tensor;

    class Setup final {
    public:
        using Run = void (*)(void* state, Tensor& out);
        using Drop = void (*)(void* state) noexcept;

        Setup() noexcept = default;

        Setup(
            void* state,
            Run run,
            Drop drop,
            std::size_t buffers = 0,
            std::size_t bytes = 0
        ) noexcept : state_(state),
            run_(run),
            drop_(drop),
            buffers_(buffers),
            bytes_(bytes) {}

        ~Setup() {
            reset();
        }

        Setup(const Setup&) = delete;
        Setup& operator=(const Setup&) = delete;

        Setup(Setup&& other) noexcept
            : state_(std::exchange(other.state_, nullptr)),
              run_(std::exchange(other.run_, nullptr)),
              drop_(std::exchange(other.drop_, nullptr)),
              buffers_(std::exchange(other.buffers_, 0)),
              bytes_(std::exchange(other.bytes_, 0)) {}

        Setup& operator=(Setup&& other) noexcept {
            if (this == &other) {
                return *this;
            }

            reset();
            state_ = std::exchange(other.state_, nullptr);
            run_ = std::exchange(other.run_, nullptr);
            drop_ = std::exchange(other.drop_, nullptr);
            buffers_ = std::exchange(other.buffers_, 0);
            bytes_ = std::exchange(other.bytes_, 0);
            return *this;
        }

        explicit operator bool() const noexcept {
            return state_ != nullptr;
        }

        void run(Tensor& out) const {
            run_(state_, out);
        }

        std::size_t buffers() const noexcept {
            return buffers_;
        }

        std::size_t bytes() const noexcept {
            return bytes_;
        }

    private:
        void reset() noexcept {
            if (state_) {
                drop_(state_);
            }
        }

        void* state_ = nullptr;
        Run run_ = nullptr;
        Drop drop_ = nullptr;
        std::size_t buffers_ = 0;
        std::size_t bytes_ = 0;
    };

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
    using Prepare = Setup (*)(Tensor&);
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
