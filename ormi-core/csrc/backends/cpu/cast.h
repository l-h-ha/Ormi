#pragma once

namespace ormi::core {
    class Tensor;
}

namespace ormi::backends::cpu {
    void cast(core::Tensor& out, const core::Tensor& input);
}
