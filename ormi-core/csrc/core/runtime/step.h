#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <ormi/core/op.h>
#include <ormi/backend/api.h>
#include <ormi/core/tensor.h>

#include "core/runtime/fusion/expr.h"

namespace ormi::core::runtime {
    class Step final {
    public:
        static Step alias(Tensor& output);
        static Step compute(
            Tensor& output,
            Compute compute,
            Setup setup = {}
        );
        static Step fusion(
            Tensor& output,
            FusionFn execute,
            fusion::Expr expression,
            std::vector<Tensor*> auxiliary_outputs = {}
        );
        static Step deferred_fusion(
            Tensor& output,
            PrepareFusionFn prepare,
            PrepareFusionsFn prepare_many,
            FusionFn execute,
            fusion::Expr expression,
            std::vector<Tensor*> auxiliary_outputs = {}
        );

        Step(const Step&) = delete;
        Step& operator=(const Step&) = delete;
        Step(Step&&) noexcept = default;
        Step& operator=(Step&&) noexcept = default;

        Tensor& output() const noexcept;
        std::size_t output_count() const noexcept;
        Tensor& output(std::size_t index) const noexcept;
        std::size_t input_count() const noexcept;
        Tensor& input(std::size_t index) const noexcept;
        bool allocates() const noexcept;
        std::size_t setup_buffers() const noexcept;
        std::size_t setup_bytes() const noexcept;
        void prepare();
        void run();

    private:
        friend void prepare(std::span<Step> steps);

        enum class Kind : std::uint8_t {
            alias,
            compute,
            fusion
        };

        enum class FusionState : std::uint8_t {
            direct,
            pending,
            prepared
        };

        explicit Step(Tensor& output) noexcept;
        Step(Tensor& output, Compute compute, Setup setup) noexcept;
        Step(
            Tensor& output,
            FusionFn execute,
            fusion::Expr expression,
            std::vector<Tensor*> auxiliary_outputs
        ) noexcept;

        FusionJob job() noexcept;
        void prepared() noexcept;

        Tensor* output_;
        Kind kind_;
        FusionState fusion_state_ = FusionState::direct;
        Compute compute_ = nullptr;
        Setup setup_;
        PrepareFusionFn fusion_prepare_ = nullptr;
        PrepareFusionsFn fusion_prepare_many_ = nullptr;
        FusionFn fusion_ = nullptr;
        std::optional<fusion::Expr> expression_;
        std::vector<Tensor*> auxiliary_outputs_;
    };

    void prepare(std::span<Step> steps);

}
