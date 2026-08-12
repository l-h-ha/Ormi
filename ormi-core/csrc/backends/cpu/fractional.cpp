#include <ormi/core/attrs.h>
#include <ormi/core/ops/kind.h>
#include <ormi/core/tensor.h>

#include "backends/cpu/fractional.h"
#include "backends/cpu/fractional/l1.h"
#include "backends/cpu/fractional/l2.h"

namespace ormi::backends::cpu {
    void fractional(
        core::ops::FractionalOp op,
        core::Tensor& out,
        const core::Tensor& input,
        const core::Tensor* initial_derivative,
        const core::Caputo& attrs
    ) {
        if (attrs.alpha < 1.0) {
            caputo::l1(op, out, input, attrs);
            return;
        }
        caputo::l2(op, out, input, initial_derivative, attrs);
    }
}
