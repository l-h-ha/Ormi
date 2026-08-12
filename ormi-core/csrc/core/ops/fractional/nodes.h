#pragma once
#include <memory>

#include <ormi/core/attrs.h>

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::ops::fractional {
    std::shared_ptr<Tensor> l1(
        const std::shared_ptr<Tensor>& input,
        Caputo attrs
    );

    std::shared_ptr<Tensor> l2(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& initial_derivative,
        Caputo attrs
    );

    bool is_node(const Tensor& tensor);
}
