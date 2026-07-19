#include <ormi/core/runtime/backend.h>
#include <ormi/core/runtime/dispatch/optimizer.h>
#include <ormi/core/tensor.h>

namespace ormi::core::runtime::optimizer {
    void sgd(
        Tensor& out,
        const Tensor& parameter,
        const Tensor& gradient,
        double step_size
    ) {
        out.device().backend()->sgd(out, parameter, gradient, step_size);
    }
}
