#pragma once

#include <memory>

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::optim {
    std::shared_ptr<Tensor> sgd(
        const std::shared_ptr<Tensor>& parameter,
        const std::shared_ptr<Tensor>& gradient,
        double learning_rate,
        double gradient_scale = 1.0
    );
}
