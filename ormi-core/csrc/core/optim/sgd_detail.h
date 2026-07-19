#pragma once
#include <memory>

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::optim::detail {
    double check_sgd(
        const std::shared_ptr<Tensor>& parameter,
        const std::shared_ptr<Tensor>& gradient,
        double learning_rate,
        double gradient_scale
    );

    std::shared_ptr<Tensor> create_sgd(
        const std::shared_ptr<Tensor>& parameter,
        const std::shared_ptr<Tensor>& gradient,
        double step_size
    );
}
