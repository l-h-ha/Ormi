#include <cmath>
#include <memory>

#include <ormi/core/error.h>
#include <ormi/core/ops/pointwise.h>
#include <ormi/nn/activation.h>
#include <ormi/nn/modules/activation.h>

namespace ormi::nn {
    std::shared_ptr<core::Tensor> ReLU::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::relu(input);
    }

    std::shared_ptr<core::Tensor> Sigmoid::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::sigmoid(input);
    }

    std::shared_ptr<core::Tensor> Tanh::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::tanh(input);
    }

    std::shared_ptr<core::Tensor> Softplus::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::softplus(input);
    }

    std::shared_ptr<core::Tensor> GELU::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::gelu(input);
    }

    std::shared_ptr<core::Tensor> SiLU::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::silu(input);
    }

    std::shared_ptr<core::Tensor> Softsign::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::softsign(input);
    }

    std::shared_ptr<core::Tensor> Mish::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::mish(input);
    }

    std::shared_ptr<core::Tensor> Softmax::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::softmax(input, axis_);
    }

    std::shared_ptr<core::Tensor> LogSoftmax::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::log_softmax(input, axis_);
    }

    LeakyReLU::LeakyReLU(double negative_slope)
        : negative_slope_(negative_slope) {
        if (!std::isfinite(negative_slope) || negative_slope < 0) {
            ORMI_THROW(OperationError, "LeakyReLU negative_slope must be finite and nonnegative.");
        }
    }

    std::shared_ptr<core::Tensor> LeakyReLU::forward(const std::shared_ptr<core::Tensor>& input) const {
        return nn::leaky_relu(input, negative_slope_);
    }

    std::shared_ptr<core::Tensor> Sin::forward(const std::shared_ptr<core::Tensor>& input) const {
        return core::ops::sin(input);
    }
}
