#pragma once
#include <memory>

#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs);
    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs);
    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs);
    std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs);

    std::shared_ptr<Tensor> neg(const std::shared_ptr<Tensor>& t);
    std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> sin(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> cos(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> abs(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> sign(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> positive(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> softplus(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> erf(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> log1p(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> expm1(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> rsqrt(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> minimum(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs);
    std::shared_ptr<Tensor> maximum(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs);
}
