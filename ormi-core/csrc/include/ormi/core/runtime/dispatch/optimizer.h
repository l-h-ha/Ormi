#pragma once

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::runtime::optimizer {
    void sgd(
        Tensor& out,
        const Tensor& parameter,
        const Tensor& gradient,
        double step_size
    );
}
