#pragma once

namespace ormi::core {
    class Tensor;
}

namespace ormi::backends::cuda {
    void sgd(
        core::Tensor& out,
        const core::Tensor& parameter,
        const core::Tensor& gradient,
        double step_size
    );
}
