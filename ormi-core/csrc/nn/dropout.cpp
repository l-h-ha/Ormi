#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <variant>
#include <vector>

#include <ormi/core/autograd.h>
#include <ormi/core/memory/ops.h>
#include <ormi/core/ops/cast.h>
#include <ormi/core/ops/creation.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/nn/dropout.h>
#include <ormi/nn/modules/dropout.h>

#include "core/runtime/device.h"
#include "nn/functional.h"

namespace ormi::nn {
    namespace {
        void check_probability(double p) {
            if (!std::isfinite(p) || p < 0 || p > 1) {
                ORMI_THROW(OperationError, "Dropout p must be finite and between zero and one.");
            }
        }

        std::uint64_t random_seed() {
            std::random_device random;
            return (static_cast<std::uint64_t>(random()) << 32) | random();
        }

        std::shared_ptr<core::memory::Buffer> make_counter(core::Device device) {
            auto counter = std::make_shared<core::memory::Buffer>(device, sizeof(std::uint64_t));
            core::memory::zero(*counter);
            return counter;
        }

        void compute_mask(core::Tensor& output) {
            const auto& attrs = std::get<core::Bernoulli>(output.attrs());
            core::runtime::backend(output.device()).ops.bernoulli(output, attrs);
        }

        // The random choice is not differentiated. The multiply below supplies input gradients
        // using this same mask node, so backward must not request another draw.
        void mask_adjoint(
            const std::shared_ptr<core::Tensor>&,
            const std::shared_ptr<core::Tensor>&,
            core::Grads&
        ) {}

        constexpr core::Op mask{
            .name = "bernoulli",
            .output = core::Output::allocate,
            .parents = 1,
            .compute = compute_mask,
            .adjoint = mask_adjoint,
            .scalar = {}
        };

        std::shared_ptr<core::Tensor> apply_dropout(
            const std::shared_ptr<core::Tensor>& input,
            double p,
            std::uint64_t seed,
            const std::shared_ptr<core::memory::Buffer>& counter
        ) {
            const auto value = detail::floating(input);
            auto draws = std::make_shared<core::Tensor>(
                core::op_ref<mask>(),
                core::Bernoulli{p, seed, counter},
                std::vector{value},
                value->device(),
                value->shape(),
                value->dtype()
            );
            return core::ops::cast(core::ops::mul(value, draws), input->dtype());
        }
    }

    std::shared_ptr<core::Tensor> dropout(
        const std::shared_ptr<core::Tensor>& input,
        double p,
        bool training,
        std::optional<std::uint64_t> seed
    ) {
        check_probability(p);
        detail::floating(input);
        if (!training || p == 0) {
            return input;
        }
        if (p == 1) {
            return core::ops::zeros_like(input);
        }
        return apply_dropout(input, p, seed ? *seed : random_seed(), make_counter(input->device()));
    }

    Dropout::Dropout(double p, std::optional<std::uint64_t> seed)
        : p_(p),
          seed_(seed ? *seed : random_seed()) {
        check_probability(p_);
    }

    std::shared_ptr<core::Tensor> Dropout::forward(const std::shared_ptr<core::Tensor>& input) const {
        detail::floating(input);
        if (!training() || p_ == 0) {
            return input;
        }
        if (p_ == 1) {
            return core::ops::zeros_like(input);
        }
        for (const auto& counter : counters_) {
            if (counter->device() == input->device()) {
                return apply_dropout(input, p_, seed_, counter);
            }
        }
        auto counter = make_counter(input->device());
        counters_.push_back(counter);
        return apply_dropout(input, p_, seed_, counter);
    }

    double Dropout::p() const noexcept {
        return p_;
    }

    std::uint64_t Dropout::seed() const noexcept {
        return seed_;
    }
}
