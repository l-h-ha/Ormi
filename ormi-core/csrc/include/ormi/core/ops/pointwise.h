#pragma once
#include <memory>

#include <ormi/core/tensor.h>

namespace ormi::core::ops {
    std::shared_ptr<Tensor> add(
        const std::shared_ptr<Tensor>& lhs,
        const std::shared_ptr<Tensor>& rhs
    );
    std::shared_ptr<Tensor> sub(
        const std::shared_ptr<Tensor>& lhs,
        const std::shared_ptr<Tensor>& rhs
    );
    std::shared_ptr<Tensor> mul(
        const std::shared_ptr<Tensor>& lhs,
        const std::shared_ptr<Tensor>& rhs
    );
    std::shared_ptr<Tensor> div(
        const std::shared_ptr<Tensor>& lhs,
        const std::shared_ptr<Tensor>& rhs
    );

    std::shared_ptr<Tensor> neg(const std::shared_ptr<Tensor>& t);
    std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> sin(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> cos(const std::shared_ptr<Tensor>& input);
}
