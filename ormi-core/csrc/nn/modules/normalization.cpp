#include <memory>
#include <utility>

#include <ormi/core/float16.h>
#include <ormi/core/memory/ops.h>
#include <ormi/nn/modules/normalization.h>
#include <ormi/nn/normalization.h>

#include "nn/functional.h"

namespace ormi::nn {
    LayerNorm::LayerNorm(
        core::Shape normalized_shape,
        double eps,
        bool elementwise_affine,
        bool bias,
        core::DType dtype,
        core::Device device
    )
        : normalized_shape_(std::move(normalized_shape)),
          eps_(eps) {
        detail::check_normalization(normalized_shape_, eps_);
        if (!core::is_floating(dtype)) {
            ORMI_THROW(DTypeError, "LayerNorm parameters require a floating-point dtype.");
        }
        if (!elementwise_affine) {
            return;
        }

        const double one64 = 1.0;
        const float one32 = 1.0f;
        const core::Float16 one16{1.0f};
        const void* one = &one32;

        if (dtype == core::DType::float64) {
            one = &one64;
        } else if (dtype == core::DType::float16) {
            one = &one16;
        }

        weight_ = std::make_shared<core::Tensor>(device, normalized_shape_, dtype);
        core::memory::fill_scalar(weight_->data(), device, one, weight_->numel(), dtype);
        register_parameter(weight_, "weight");
        if (!bias) {
            return;
        }

        bias_ = std::make_shared<core::Tensor>(device, normalized_shape_, dtype);
        core::memory::zero(*bias_->buffer());
        register_parameter(bias_, "bias");
    }

    std::shared_ptr<core::Tensor> LayerNorm::forward(const std::shared_ptr<core::Tensor>& input) const {
        return layer_norm(input, normalized_shape_, weight_, bias_, eps_);
    }

    const core::Shape& LayerNorm::normalized_shape() const noexcept {
        return normalized_shape_;
    }

    double LayerNorm::eps() const noexcept {
        return eps_;
    }

    std::shared_ptr<core::Tensor> LayerNorm::weight() const noexcept {
        return weight_;
    }

    std::shared_ptr<core::Tensor> LayerNorm::bias() const noexcept {
        return bias_;
    }
}
