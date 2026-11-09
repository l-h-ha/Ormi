#pragma once
#include <memory>

#include <ormi/core/device.h>
#include <ormi/core/dtype.h>
#include <ormi/core/shape.h>
#include <ormi/core/tensor.h>
#include <ormi/nn/modules/module.h>

namespace ormi::nn {
    // Normalize the trailing axes that match normalized_shape, leaving leading axes unchanged.
    // Weights start at one and biases at zero. Float16 statistics use Float32.
    class LayerNorm final : public Module {
    public:
        explicit LayerNorm(
            core::Shape normalized_shape,
            double eps = 1e-5,
            bool elementwise_affine = true,
            bool bias = true,
            core::DType dtype = core::DType::float32,
            core::Device device = core::Device::cpu()
        );

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) const override;

        const core::Shape& normalized_shape() const noexcept;
        double eps() const noexcept;
        std::shared_ptr<core::Tensor> weight() const noexcept;
        std::shared_ptr<core::Tensor> bias() const noexcept;

    private:
        core::Shape normalized_shape_;
        double eps_;
        std::shared_ptr<core::Tensor> weight_;
        std::shared_ptr<core::Tensor> bias_;
    };
}
