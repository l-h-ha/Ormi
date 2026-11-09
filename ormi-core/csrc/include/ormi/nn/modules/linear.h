#pragma once
#include <cstdint>
#include <memory>
#include <optional>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/tensor.h>
#include <ormi/nn/modules/module.h>

namespace ormi::nn {
    class Linear final : public Module {
    public:
        Linear(
            std::int64_t in_features,
            std::int64_t out_features,
            bool bias = true,
            core::DType dtype = core::DType::float32,
            core::Device device = core::Device::cpu(),
            std::optional<std::uint64_t> seed = std::nullopt
        );

        // Weight has shape [in_features, out_features]. Tensors must be realized leaves.
        // Retains the supplied Tensor objects so modules can share parameters.
        explicit Linear(std::shared_ptr<core::Tensor> weight, std::shared_ptr<core::Tensor> bias = nullptr);

        // [..., in_features] @ weight + optional bias -> [..., out_features].
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;

        std::int64_t in_features() const noexcept;
        std::int64_t out_features() const noexcept;
        std::shared_ptr<core::Tensor> weight() const noexcept;
        std::shared_ptr<core::Tensor> bias() const noexcept;

    private:
        std::shared_ptr<core::Tensor> weight_;
        std::shared_ptr<core::Tensor> bias_;
    };
}
