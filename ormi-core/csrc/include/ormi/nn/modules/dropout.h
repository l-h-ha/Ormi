#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <ormi/core/memory/buffer.h>
#include <ormi/core/tensor.h>
#include <ormi/nn/modules/module.h>

namespace ormi::nn {
    // Drop elements with probability p and scale retained values by 1/(1-p).
    // Forward calls share an execution counter per device. Repeated Plans and
    // recordings draw a fresh mask. Derivatives reuse that mask.
    // Seed reproducibility requires the same backend/build and execution order.
    class Dropout final : public Module {
    public:
        explicit Dropout(double p = 0.5, std::optional<std::uint64_t> seed = std::nullopt);
        // Evaluation or p=0 returns the input itself. p=1 builds a zero-valued result.
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;
        double p() const noexcept;
        std::uint64_t seed() const noexcept;

    private:
        double p_;
        std::uint64_t seed_;
        mutable std::vector<std::shared_ptr<core::memory::Buffer>> counters_;
    };
}
