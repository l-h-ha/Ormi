#pragma once

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::runtime::cast {
    void run(Tensor& out, const Tensor& input);
}
