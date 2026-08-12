#pragma once

namespace ormi::core {
    class Tensor;
}

namespace ormi::core::runtime {
    void realize(Tensor& tensor);
}
