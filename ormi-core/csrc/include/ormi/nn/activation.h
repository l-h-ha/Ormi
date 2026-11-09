#pragma once
#include <cstdint>
#include <memory>

#include <ormi/core/tensor.h>

namespace ormi::nn {
    std::shared_ptr<core::Tensor> relu(const std::shared_ptr<core::Tensor>& input);
    std::shared_ptr<core::Tensor> sigmoid(const std::shared_ptr<core::Tensor>& input);
    std::shared_ptr<core::Tensor> tanh(const std::shared_ptr<core::Tensor>& input);
    std::shared_ptr<core::Tensor> softplus(const std::shared_ptr<core::Tensor>& input);
    std::shared_ptr<core::Tensor> leaky_relu(
        const std::shared_ptr<core::Tensor>& input,
        double negative_slope = 0.01
    );
    std::shared_ptr<core::Tensor> gelu(const std::shared_ptr<core::Tensor>& input);
    std::shared_ptr<core::Tensor> silu(const std::shared_ptr<core::Tensor>& input);
    std::shared_ptr<core::Tensor> softsign(const std::shared_ptr<core::Tensor>& input);
    std::shared_ptr<core::Tensor> mish(const std::shared_ptr<core::Tensor>& input);
    std::shared_ptr<core::Tensor> softmax(const std::shared_ptr<core::Tensor>& input, std::int64_t axis = -1);
    std::shared_ptr<core::Tensor> log_softmax(
        const std::shared_ptr<core::Tensor>& input,
        std::int64_t axis = -1
    );
}
